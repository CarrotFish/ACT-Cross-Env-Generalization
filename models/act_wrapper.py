"""
ACT (Action Chunking with Transformers) 模型封装器

本模块对 LeRobot 框架中集成的 ACT 算法进行封装，提供统一的接口：
  - 模型构建 (从配置文件初始化)
  - 前向推理 (训练模式 / 推理模式)
  - 损失计算 (Action L1 Loss + KL Divergence Loss)
  - 权重加载与保存

ACT 架构概述:
  ┌─────────────────────────────────────────────────────────┐
  │                     ACT 模型                            │
  │                                                         │
  │  图像观测 ──► ResNet 视觉编码器 ──► 视觉特征 Token       │
  │                                          │              │
  │  机器人状态 ──────────────────────────► Transformer     │
  │                                       解码器            │
  │  (训练时) 动作序列 ──► CVAE 编码器 ──► 潜变量 z ──►     │
  │  (推理时) z ~ N(0,1) ──────────────────────────────►   │
  │                                          │              │
  │                              预测动作序列 (chunk_size,   │
  │                                          action_dim)    │
  └─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path


# ============================================================
# CVAE 编码器 (训练时编码动作序列为潜变量)
# ============================================================

class CVAEEncoder(nn.Module):
    """
    条件变分自编码器 (CVAE) 编码器。
    将动作序列和机器人状态编码为潜变量分布 (mu, log_var)。

    输入:
      - actions: (B, chunk_size, action_dim)
      - state:   (B, state_dim)
    输出:
      - mu:      (B, latent_dim)
      - log_var: (B, latent_dim)
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        latent_dim: int,
        chunk_size: int,
        nheads: int = 8,
        enc_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 动作序列线性投影
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # 状态线性投影
        self.state_proj  = nn.Linear(state_dim, hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )

        # 位置编码 (可学习)
        self.pos_embedding = nn.Embedding(chunk_size + 1, hidden_dim)

        # 输出 mu 和 log_var
        self.mu_proj      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        actions: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            actions: (B, chunk_size, action_dim)
            state:   (B, state_dim)

        Returns:
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
        """
        B, T, _ = actions.shape

        # 投影动作序列: (B, T, hidden_dim)
        action_tokens = self.action_proj(actions)

        # 投影状态并作为 CLS token: (B, 1, hidden_dim)
        state_token = self.state_proj(state).unsqueeze(1)

        # 拼接: (B, T+1, hidden_dim)
        tokens = torch.cat([state_token, action_tokens], dim=1)

        # 添加位置编码
        positions = torch.arange(T + 1, device=tokens.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)

        # Transformer 编码
        encoded = self.transformer_encoder(tokens)  # (B, T+1, hidden_dim)

        # 取 CLS token (第 0 位) 作为序列表示
        cls_token = encoded[:, 0, :]  # (B, hidden_dim)

        mu      = self.mu_proj(cls_token)       # (B, latent_dim)
        log_var = self.log_var_proj(cls_token)  # (B, latent_dim)

        return mu, log_var


# ============================================================
# 视觉编码器 (ResNet backbone)
# ============================================================

class VisualEncoder(nn.Module):
    """
    基于 ResNet-18 的视觉特征提取器。
    将 RGB 图像编码为特征 Token 序列，供 Transformer 解码器使用。

    输入:  (B, C, H, W) 图像
    输出:  (B, num_patches, hidden_dim) 特征序列
    """

    def __init__(self, hidden_dim: int, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv_models

        # 使用 ResNet-18 作为 backbone
        backbone = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # 去掉最后的全连接层和平均池化，保留空间特征图
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # ResNet-18 最后一层输出通道数为 512
        self.feature_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W)

        Returns:
            features: (B, H'*W', hidden_dim) 展平的空间特征序列
        """
        feat_map = self.backbone(image)           # (B, 512, H', W')
        feat_map = self.feature_proj(feat_map)    # (B, hidden_dim, H', W')

        B, C, H, W = feat_map.shape
        # 展平空间维度: (B, H'*W', hidden_dim)
        features = feat_map.flatten(2).permute(0, 2, 1)

        return features


# ============================================================
# ACT 主模型
# ============================================================

class ACTModel(nn.Module):
    """
    ACT (Action Chunking with Transformers) 核心模型。

    训练流程:
      1. CVAE 编码器将 (actions, state) 编码为潜变量 (mu, log_var)
      2. 重参数化采样得到 z
      3. 视觉编码器提取图像特征
      4. Transformer 解码器以 (z, state, visual_features) 为条件，
         预测 chunk_size 步的动作序列

    推理流程:
      1. 从标准正态分布采样 z ~ N(0, I)
      2. 其余步骤与训练相同
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        latent_dim: int,
        chunk_size: int,
        nheads: int,
        enc_layers: int,
        dec_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_vae: bool = True,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim
        self.chunk_size  = chunk_size
        self.use_vae     = use_vae

        # --- 视觉编码器 ---
        self.visual_encoder = VisualEncoder(hidden_dim=hidden_dim, pretrained=True)

        # --- CVAE 编码器 (仅训练时使用) ---
        if use_vae:
            self.cvae_encoder = CVAEEncoder(
                action_dim=action_dim,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                chunk_size=chunk_size,
                nheads=nheads,
                enc_layers=enc_layers,
                dropout=dropout,
            )

        # --- 状态与潜变量投影 ---
        self.state_proj   = nn.Linear(state_dim, hidden_dim)
        self.latent_proj  = nn.Linear(latent_dim, hidden_dim)

        # --- 动作查询 (可学习的 Query Tokens) ---
        self.action_queries = nn.Embedding(chunk_size, hidden_dim)

        # --- Transformer 解码器 ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dec_layers
        )

        # --- 动作预测头 ---
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        重参数化技巧: z = mu + eps * std, eps ~ N(0, I)

        Args:
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)

        Returns:
            z: (B, latent_dim)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            image:   (B, C, H, W) 图像观测
            state:   (B, state_dim) 机器人状态
            actions: (B, chunk_size, action_dim) 目标动作序列 (训练时提供，推理时为 None)

        Returns:
            字典包含:
              - "pred_actions": (B, chunk_size, action_dim) 预测动作序列
              - "mu":           (B, latent_dim) CVAE 均值 (训练时)
              - "log_var":      (B, latent_dim) CVAE 对数方差 (训练时)
        """
        B = image.shape[0]
        device = image.device

        # 1. 视觉特征提取: (B, num_patches, hidden_dim)
        visual_features = self.visual_encoder(image)

        # 2. 潜变量处理
        if self.use_vae and actions is not None:
            # 训练模式: 通过 CVAE 编码器获取潜变量分布
            mu, log_var = self.cvae_encoder(actions, state)
            z = self.reparameterize(mu, log_var)
        else:
            # 推理模式: 从标准正态分布采样
            z = torch.zeros(B, self.latent_dim, device=device)
            mu = z
            log_var = z

        # 3. 构建解码器的 Memory (条件信息)
        # 状态 token: (B, 1, hidden_dim)
        state_token   = self.state_proj(state).unsqueeze(1)
        # 潜变量 token: (B, 1, hidden_dim)
        latent_token  = self.latent_proj(z).unsqueeze(1)
        # 拼接所有条件 token: (B, num_patches + 2, hidden_dim)
        memory = torch.cat([latent_token, state_token, visual_features], dim=1)

        # 4. 动作查询 tokens: (B, chunk_size, hidden_dim)
        query_pos = torch.arange(self.chunk_size, device=device)
        queries   = self.action_queries(query_pos).unsqueeze(0).expand(B, -1, -1)

        # 5. Transformer 解码
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=memory,
        )  # (B, chunk_size, hidden_dim)

        # 6. 预测动作序列
        pred_actions = self.action_head(decoded)  # (B, chunk_size, action_dim)

        return {
            "pred_actions": pred_actions,
            "mu":           mu,
            "log_var":      log_var,
        }


# ============================================================
# ACT 封装器 (含损失计算与权重管理)
# ============================================================

class ACTWrapper(nn.Module):
    """
    ACT 模型的高级封装器，提供：
      - 统一的训练/推理接口
      - Action L1 Loss + KL Divergence Loss 计算
      - 模型权重的保存与加载
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: 完整配置字典 (来自 base_act.yaml)
        """
        super().__init__()
        model_cfg = cfg.get("model", {})
        loss_cfg  = cfg.get("loss", {})

        self.model = ACTModel(
            state_dim       = model_cfg.get("state_dim", 7),
            action_dim      = model_cfg.get("action_dim", 7),
            hidden_dim      = model_cfg.get("hidden_dim", 512),
            latent_dim      = model_cfg.get("latent_dim", 32),
            chunk_size      = model_cfg.get("chunk_size", 100),
            nheads          = model_cfg.get("nheads", 8),
            enc_layers      = model_cfg.get("enc_layers", 4),
            dec_layers      = model_cfg.get("dec_layers", 7),
            dim_feedforward = model_cfg.get("dim_feedforward", 3200),
            dropout         = model_cfg.get("dropout", 0.1),
            use_vae         = model_cfg.get("use_vae", True),
        )

        self.kl_weight          = loss_cfg.get("kl_loss_weight", 10.0)
        self.action_loss_weight = loss_cfg.get("action_loss_weight", 1.0)

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失。

        Args:
            pred_actions:   (B, chunk_size, action_dim) 预测动作
            target_actions: (B, chunk_size, action_dim) 目标动作
            mu:             (B, latent_dim) CVAE 均值
            log_var:        (B, latent_dim) CVAE 对数方差

        Returns:
            字典包含:
              - "total_loss":  总损失
              - "action_loss": Action L1 Loss
              - "kl_loss":     KL 散度损失
        """
        # Action L1 Loss (逐元素绝对误差均值)
        action_loss = F.l1_loss(pred_actions, target_actions, reduction="mean")

        # KL 散度损失: KL(N(mu, sigma) || N(0, I))
        # = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

        total_loss = (
            self.action_loss_weight * action_loss
            + self.kl_weight * kl_loss
        )

        return {
            "total_loss":  total_loss,
            "action_loss": action_loss,
            "kl_loss":     kl_loss,
        }

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 (代理到 ACTModel)。

        Args:
            image:   (B, C, H, W)
            state:   (B, state_dim)
            actions: (B, chunk_size, action_dim) 训练时提供

        Returns:
            模型输出字典
        """
        return self.model(image, state, actions)

    def save_checkpoint(self, path: str, epoch: int, optimizer_state: dict = None):
        """
        保存模型权重与训练状态。

        Args:
            path:            保存路径 (.pth 文件)
            epoch:           当前训练轮数
            optimizer_state: 优化器状态字典 (可选)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch":       epoch,
            "model_state": self.state_dict(),
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state"] = optimizer_state

        torch.save(checkpoint, path)
        print(f"[Checkpoint] 模型已保存至: {path} (epoch={epoch})")

    def load_checkpoint(self, path: str, device: str = "cpu") -> int:
        """
        加载模型权重。

        Args:
            path:   权重文件路径
            device: 加载设备

        Returns:
            已训练的 epoch 数
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint.get("epoch", 0)
        print(f"[Checkpoint] 已从 {path} 加载模型权重 (epoch={epoch})")
        return epoch


# ============================================================
# 模型构建工厂函数
# ============================================================

def build_model(cfg: dict) -> ACTWrapper:
    """
    根据配置字典构建 ACTWrapper 模型。

    Args:
        cfg: 完整配置字典

    Returns:
        ACTWrapper 实例
    """
    model = ACTWrapper(cfg)

    # 统计参数量
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ACT 模型构建完成")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    return model
