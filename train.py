"""
ACT 跨环境泛化训练入口

支持:
  - 单环境训练 (Baseline): python train.py --config configs/train_single_env.yaml
  - 多环境联合训练:        python train.py --config configs/train_joint_env.yaml
  - 从断点恢复训练:        python train.py --config configs/train_joint_env.yaml --resume checkpoints/joint_env_ABC/epoch_50.pth

训练流程:
  1. 加载配置文件 (YAML)
  2. 构建数据集与 DataLoader
  3. 构建 ACT 模型
  4. 初始化优化器与学习率调度器
  5. 训练循环 (含 WandB/SwanLab 日志记录)
  6. 定期保存模型权重
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from datasets import build_dataloader
from models   import build_model
from utils    import build_logger, AverageMeter


# ============================================================
# 工具函数
# ============================================================

def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件，支持 defaults 继承机制。

    Args:
        config_path: 配置文件路径

    Returns:
        合并后的配置字典
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 处理 defaults 继承
    if "defaults" in cfg:
        base_configs = cfg.pop("defaults")
        merged_cfg = {}
        for base_name in base_configs:
            base_path = config_path.parent / f"{base_name}.yaml"
            with open(base_path, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f)
            merged_cfg = deep_merge(merged_cfg, base_cfg)
        # 子配置覆盖基础配置
        cfg = deep_merge(merged_cfg, cfg)

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并两个字典，override 中的值覆盖 base 中的值。

    Args:
        base:     基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed: int):
    """设置全局随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_optimizer(model: torch.nn.Module, cfg: dict) -> optim.Optimizer:
    """
    根据配置构建优化器。

    Args:
        model: 模型
        cfg:   完整配置字典

    Returns:
        优化器实例
    """
    train_cfg = cfg.get("training", {})
    opt_cfg   = cfg.get("optimizer", {})

    lr           = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    opt_name     = opt_cfg.get("name", "AdamW")
    betas        = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps          = opt_cfg.get("eps", 1e-8)

    if opt_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif opt_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    else:
        raise ValueError(f"不支持的优化器: {opt_name}")

    print(f"[Optimizer] {opt_name} | lr={lr} | weight_decay={weight_decay}")
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, total_steps: int):
    """
    根据配置构建学习率调度器。

    Args:
        optimizer:   优化器
        cfg:         完整配置字典
        total_steps: 总训练步数

    Returns:
        学习率调度器实例
    """
    train_cfg    = cfg.get("training", {})
    scheduler_type = train_cfg.get("lr_scheduler", "cosine")
    warmup_steps   = train_cfg.get("warmup_steps", 1000)

    if scheduler_type == "cosine":
        # 余弦退火调度器 (含线性预热)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return scheduler


# ============================================================
# 单个 Epoch 训练
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch: int,
    cfg: dict,
    logger,
    global_step: int,
) -> tuple:
    """
    执行单个 epoch 的训练。

    Args:
        model:       ACTWrapper 模型
        dataloader:  训练数据加载器
        optimizer:   优化器
        scheduler:   学习率调度器
        scaler:      混合精度 GradScaler
        device:      训练设备
        epoch:       当前 epoch 编号
        cfg:         配置字典
        logger:      日志记录器
        global_step: 全局训练步数

    Returns:
        (epoch_metrics, global_step): epoch 平均指标字典和更新后的全局步数
    """
    model.train()

    train_cfg    = cfg.get("training", {})
    log_cfg      = cfg.get("logging", {})
    use_amp      = cfg.get("hardware", {}).get("use_amp", True)
    grad_clip    = train_cfg.get("grad_clip_norm", 10.0)
    log_interval = log_cfg.get("log_interval", 50)

    # 初始化指标统计器
    meters = {
        "total_loss":  AverageMeter("train/total_loss"),
        "action_loss": AverageMeter("train/action_loss"),
        "kl_loss":     AverageMeter("train/kl_loss"),
    }

    for batch_idx, batch in enumerate(dataloader):
        # 将数据移至设备
        image   = batch["image"].to(device, non_blocking=True)
        state   = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        B = image.shape[0]

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(enabled=use_amp):
            outputs = model(image, state, actions)
            losses  = model.compute_loss(
                pred_actions   = outputs["pred_actions"],
                target_actions = actions,
                mu             = outputs["mu"],
                log_var        = outputs["log_var"],
            )

        # 反向传播
        scaler.scale(losses["total_loss"]).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 更新统计器
        for key in meters:
            meters[key].update(losses[key].item(), n=B)

        global_step += 1

        # 定期记录日志
        if global_step % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            log_metrics = {
                "train/total_loss":  meters["total_loss"].val,
                "train/action_loss": meters["action_loss"].val,
                "train/kl_loss":     meters["kl_loss"].val,
                "train/lr":          current_lr,
                "train/epoch":       epoch,
            }
            logger.log(log_metrics, step=global_step)

            print(
                f"Epoch [{epoch:3d}] Step [{batch_idx+1:4d}/{len(dataloader)}] "
                f"Loss: {meters['total_loss'].val:.4f} "
                f"(action={meters['action_loss'].val:.4f}, kl={meters['kl_loss'].val:.4f}) "
                f"LR: {current_lr:.2e}"
            )

    # 返回 epoch 平均指标
    epoch_metrics = {
        "train/epoch_total_loss":  meters["total_loss"].avg,
        "train/epoch_action_loss": meters["action_loss"].avg,
        "train/epoch_kl_loss":     meters["kl_loss"].avg,
    }

    return epoch_metrics, global_step


# ============================================================
# 验证
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    epoch: int,
    cfg: dict,
) -> dict:
    """
    在验证集上评估模型的 Action L1 Loss。

    Args:
        model:      ACTWrapper 模型
        dataloader: 验证数据加载器
        device:     评估设备
        epoch:      当前 epoch
        cfg:        配置字典

    Returns:
        验证指标字典
    """
    model.eval()
    use_amp = cfg.get("hardware", {}).get("use_amp", True)

    meters = {
        "total_loss":  AverageMeter("val/total_loss"),
        "action_loss": AverageMeter("val/action_loss"),
        "kl_loss":     AverageMeter("val/kl_loss"),
    }

    for batch in dataloader:
        image   = batch["image"].to(device, non_blocking=True)
        state   = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        B = image.shape[0]

        with autocast(enabled=use_amp):
            outputs = model(image, state, actions)
            losses  = model.compute_loss(
                pred_actions   = outputs["pred_actions"],
                target_actions = actions,
                mu             = outputs["mu"],
                log_var        = outputs["log_var"],
            )

        for key in meters:
            meters[key].update(losses[key].item(), n=B)

    val_metrics = {
        "val/total_loss":  meters["total_loss"].avg,
        "val/action_loss": meters["action_loss"].avg,
        "val/kl_loss":     meters["kl_loss"].avg,
    }

    print(
        f"[Val] Epoch {epoch:3d} | "
        f"Loss: {val_metrics['val/total_loss']:.4f} "
        f"(action={val_metrics['val/action_loss']:.4f}, kl={val_metrics['val/kl_loss']:.4f})"
    )

    return val_metrics


# ============================================================
# 主训练函数
# ============================================================

def train(cfg: dict, resume_path: str = None):
    """
    主训练函数。

    Args:
        cfg:         完整配置字典
        resume_path: 断点恢复路径 (可选)
    """
    # --- 基础设置 ---
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})
    hw_cfg    = cfg.get("hardware", {})
    log_cfg   = cfg.get("logging", {})
    exp_cfg   = cfg.get("experiment", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(hw_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"[Device] 使用设备: {device}")

    # --- 构建数据集 ---
    data_cfg   = cfg.get("data", {})
    train_envs = data_cfg.get("train_envs", ["A"])
    val_envs   = data_cfg.get("val_envs",   ["A"])

    print(f"[Data] 训练环境: {train_envs} | 验证环境: {val_envs}")

    train_loader = build_dataloader(cfg, env_names=train_envs, split="training",   is_train=True)
    val_loader   = build_dataloader(cfg, env_names=val_envs,   split="validation", is_train=False)

    print(f"[Data] 训练集大小: {len(train_loader.dataset)} | 验证集大小: {len(val_loader.dataset)}")

    # --- 构建模型 ---
    model = build_model(cfg).to(device)

    # --- 构建优化器与调度器 ---
    num_epochs  = train_cfg.get("num_epochs", 100)
    total_steps = num_epochs * len(train_loader)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler    = GradScaler(enabled=hw_cfg.get("use_amp", True))

    # --- 断点恢复 ---
    start_epoch = 0
    global_step = 0
    if resume_path and Path(resume_path).exists():
        start_epoch = model.load_checkpoint(resume_path, device=str(device))
        print(f"[Resume] 从 epoch {start_epoch} 继续训练")

    # --- 初始化日志记录器 ---
    exp_name = exp_cfg.get("name", "experiment")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name  = f"{exp_name}_{timestamp}"

    logger = build_logger(cfg, run_name=run_name)

    # --- 训练循环 ---
    checkpoint_dir  = paths_cfg.get("checkpoint_dir", "./checkpoints")
    save_interval   = log_cfg.get("save_interval", 10)
    best_val_loss   = float("inf")

    print(f"\n{'='*60}")
    print(f"开始训练: {run_name}")
    print(f"总 Epochs: {num_epochs} | 总 Steps: {total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        # 训练一个 epoch
        train_metrics, global_step = train_one_epoch(
            model       = model,
            dataloader  = train_loader,
            optimizer   = optimizer,
            scheduler   = scheduler,
            scaler      = scaler,
            device      = device,
            epoch       = epoch,
            cfg         = cfg,
            logger      = logger,
            global_step = global_step,
        )

        # 验证
        val_metrics = validate(model, val_loader, device, epoch, cfg)

        # 记录 epoch 级别指标
        epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
        logger.log(epoch_metrics, step=global_step)

        # 保存最优模型
        if val_metrics["val/total_loss"] < best_val_loss:
            best_val_loss = val_metrics["val/total_loss"]
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            model.save_checkpoint(best_path, epoch=epoch, optimizer_state=optimizer.state_dict())
            print(f"[Best] 新最优模型已保存 (val_loss={best_val_loss:.4f})")

        # 定期保存检查点
        if epoch % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pth")
            model.save_checkpoint(ckpt_path, epoch=epoch, optimizer_state=optimizer.state_dict())

    # --- 保存最终模型 ---
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    model.save_checkpoint(final_path, epoch=num_epochs)

    # --- 记录实验总结 ---
    logger.log_summary({
        "best_val_loss":   best_val_loss,
        "total_epochs":    num_epochs,
        "train_envs":      str(train_envs),
        "final_model_path": final_path,
    })

    logger.finish()
    print(f"\n训练完成！最优验证 Loss: {best_val_loss:.4f}")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ACT 跨环境泛化训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单环境 Baseline 训练
  python train.py --config configs/train_single_env.yaml

  # 多环境联合训练
  python train.py --config configs/train_joint_env.yaml

  # 从断点恢复
  python train.py --config configs/train_joint_env.yaml --resume checkpoints/joint_env_ABC/epoch_0050.pth

  # 覆盖配置参数
  python train.py --config configs/train_single_env.yaml --batch_size 16 --lr 5e-5
        """
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置文件路径 (e.g., configs/train_single_env.yaml)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="断点恢复路径 (e.g., checkpoints/epoch_0050.pth)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="覆盖配置中的 batch_size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="覆盖配置中的学习率"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="覆盖配置中的训练轮数"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="覆盖配置中的设备 (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help=(
            "每个环境最多使用的 episode 数量 (覆盖配置中的 data.max_episodes_per_env)。\n"
            "设备性能受限时使用，例如 --max_episodes 200 表示每个环境只用 200 条 episode。\n"
            "设为 null/不传 则使用全部数据。"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行参数覆盖配置
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr
    if args.num_epochs is not None:
        cfg.setdefault("training", {})["num_epochs"] = args.num_epochs
    if args.device is not None:
        cfg.setdefault("hardware", {})["device"] = args.device
    if args.max_episodes is not None:
        cfg.setdefault("data", {})["max_episodes_per_env"] = args.max_episodes

    # 开始训练
    train(cfg, resume_path=args.resume)
