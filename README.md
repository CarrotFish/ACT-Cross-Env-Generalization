# ACT Cross-Environment Generalization

> **复旦大学 计算机视觉 Final-PJ Task 2**  
> 基于 LeRobot 框架的 ACT (Action Chunking with Transformers) 跨环境泛化研究

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 任务概述

本项目包含三个递进式实验，研究 ACT 策略在 CALVIN 机器人操作基准上的**跨环境泛化能力**：

| 任务 | 描述 | 训练数据 | 评估环境 | 核心指标 |
|:---:|------|---------|---------|---------|
| **Task 1** | 基础策略训练 | 仅环境 **B** | 环境 B (同分布) | Action L1 Loss |
| **Task 2** | 多环境联合训练 | 环境 **A+B+C** 混合 | 环境 A/B/C (同分布) | Action L1 Loss |
| **Task 3** | Zero-shot 跨环境泛化 | — (使用 Task 1/2 的模型) | 环境 **D** (未见过) | **Success Rate** |

> **核心对比**: Task 1 模型 (仅见过 B) vs Task 2 模型 (见过 A+B+C) 在环境 D 上的 Zero-shot 性能差异，揭示多环境训练对视觉分布偏移鲁棒性的提升效果。

---

## 🏗️ 项目结构

```
ACT-Cross-Env-Generalization/
├── configs/
│   ├── base_act.yaml              # 基础超参数 (所有实验共享，通过 defaults 继承)
│   ├── train_single_env_B.yaml    # Task 1: 仅环境 B 训练
│   └── train_joint_env.yaml       # Task 2: 环境 A+B+C 联合训练
├── datasets/
│   ├── __init__.py
│   └── calvin_dataset.py          # CALVIN 数据集加载 + 多环境混合 + 数据量限制
├── models/
│   ├── __init__.py
│   └── act_wrapper.py             # ACT 模型 (CVAE + ResNet-18 + Transformer)
├── utils/
│   ├── __init__.py
│   ├── logger.py                  # WandB / SwanLab 日志封装
│   └── metrics.py                 # Success Rate + Action L1/L2 误差计算
├── scripts/
│   ├── run_train.sh               # 训练启动脚本
│   └── run_eval.sh                # 评估启动脚本
├── train.py                       # 训练入口
├── eval.py                        # 评估入口
├── environment.yml                # Conda 环境配置
└── README.md
```

---

## 🧠 模型架构

```
图像观测 (H×W×3)
    │
    ▼
ResNet-18 视觉编码器 (去掉 FC 层，保留空间特征图)
    │  → 视觉特征 Token (H'×W', hidden_dim=512)
    │
    ├── 机器人状态 (state_dim=7) → 线性投影 → State Token
    │
    ├── [训练时] 动作序列 → CVAE 编码器 → 潜变量 z (latent_dim=32)
    │   [推理时] z ~ N(0, I)
    │
    ▼
Transformer 解码器 (7层, 8头, dim=512)
    Memory = [z_token, state_token, visual_tokens]
    Query  = 可学习动作查询 (chunk_size=100)
    │
    ▼
动作预测头 → 预测动作序列 (chunk_size=100, action_dim=7)
```

### 关键超参数 (三个任务完全一致，确保对比公平)

| 参数 | 值 | 说明 |
|------|-----|------|
| **Network** | ResNet-18 + Transformer | 视觉编码器 + 序列解码器 |
| **Hidden Dim** | 512 | Transformer 嵌入维度 |
| **Encoder Layers** | 4 | CVAE Transformer 编码器层数 |
| **Decoder Layers** | 7 | 动作 Transformer 解码器层数 |
| **Attention Heads** | 8 | 多头注意力头数 |
| **Chunk Size** | 100 | Action Chunking 窗口大小 |
| **Latent Dim** | 32 | CVAE 潜变量维度 |
| **Batch Size** | 32 | 训练批大小 |
| **Learning Rate** | 1e-4 | AdamW 初始学习率 |
| **Optimizer** | AdamW | β=(0.9, 0.999), ε=1e-8 |
| **LR Scheduler** | Cosine + Warmup | 预热步数 1000 |
| **Epochs** | 100 | 训练轮数 |
| **Loss Function** | L1 Loss + KL Divergence | Action L1 + CVAE 正则化 |

---

## ⚙️ 环境配置

```bash
# 1. 克隆项目
git clone https://github.com/<your-username>/ACT-Cross-Env-Generalization.git
cd ACT-Cross-Env-Generalization

# 2. 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate act-cross-env

# 3. 验证 GPU 支持
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. 登录日志服务 (二选一)
wandb login       # WandB
# swanlab login   # SwanLab (国内推荐，需在 base_act.yaml 中设置 backend: swanlab)
```

---

## 📦 数据准备

```bash
mkdir -p data/calvin

# 从 HuggingFace 下载 CALVIN ABCD→D 数据集
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lerobot/calvin_abc_d',
    repo_type='dataset',
    local_dir='data/calvin'
)
"
```

数据集目录结构：
```
data/calvin/task_ABCD_D/
├── training/       # 环境 A, B, C 的训练数据
│   ├── ep_start_end_ids.npy
│   └── frame_*.npz
└── validation/     # 环境 D 的验证/测试数据
    ├── ep_start_end_ids.npy
    └── frame_*.npz
```

---

## 🚀 实验执行流程

### Task 1 — 基础策略训练 (仅环境 B)

```bash
# 完整数据训练
python train.py --config configs/train_single_env_B.yaml --device cuda:0

# 设备性能受限时 (减少数据量加速验证)
python train.py --config configs/train_single_env_B.yaml --device cuda:0 --max_episodes 200
```

训练完成后，模型权重保存于 `checkpoints/single_env_B/`。

---

### Task 2 — 多环境联合训练 (环境 A+B+C)

> ⚠️ 与 Task 1 使用**完全相同的网络架构和超参数**，仅训练数据不同，确保对比公平。

```bash
# 完整数据训练
python train.py --config configs/train_joint_env.yaml --device cuda:0

# 设备性能受限时
python train.py --config configs/train_joint_env.yaml --device cuda:0 --max_episodes 200
```

训练完成后，模型权重保存于 `checkpoints/joint_env_ABC/`。

---

### Task 2 补充 — 训练收敛对比

两个模型训练过程中的 Action L1 Loss 曲线将自动记录到 WandB/SwanLab。
在同一个 Project 下可直接对比两条 Loss 曲线，观察：
- 单环境模型 (B) 是否收敛更快但泛化性更差？
- 联合训练模型 (ABC) 是否收敛更慢但最终 Loss 更低？

---

### Task 3 — Zero-shot 跨环境泛化测试 (环境 D)

将 Task 1 和 Task 2 的模型分别部署到**从未见过的环境 D** 中进行测试。

#### 3a. 离线评估 (计算动作误差)

```bash
# Task 1 模型在环境 D 上的动作误差
python eval.py \
    --config configs/train_single_env_B.yaml \
    --checkpoint checkpoints/single_env_B/best_model.pth \
    --eval_envs D \
    --output_dir ./results

# Task 2 模型在环境 D 上的动作误差
python eval.py \
    --config configs/train_joint_env.yaml \
    --checkpoint checkpoints/joint_env_ABC/best_model.pth \
    --eval_envs D \
    --output_dir ./results
```

评估结果将自动保存为 JSON 文件至 `./results/`，并在终端打印如下格式的汇总表：

```
评估环境: D
───────────────────────────────────────────
  L1 Error: 0.0523 | L2 Error: 0.0089 | Samples: 1000
  Simulated Success Rate (Strict): 45.2% | (Loose): 72.8%
  Action Error Percentiles - Median: 0.0312, P90: 0.0856, P95: 0.1234, P99: 0.1876
```

**输出指标说明**:
- **L1 Error**: 预测动作与真实动作的平均 L1 范数误差
- **L2 Error**: 预测动作与真实动作的平均 L2 范数误差 (MSE)
- **Simulated Success Rate (Strict)**: 基于严格阈值 (位置<5cm, 角度<5°) 的模拟成功率
- **Simulated Success Rate (Loose)**: 基于宽松阈值 (位置<10cm, 角度<10°) 的模拟成功率
- **Action Error Percentiles**: 动作误差的分位数统计 (中位数, 90/95/99 百分位)

---
