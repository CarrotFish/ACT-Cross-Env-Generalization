"""
评估指标计算模块

提供：
  - 成功率 (Success Rate) 计算 - CALVIN 环境的核心评估指标
  - 动作误差 (Action L1/L2 Error) 计算
  - AverageMeter - 训练过程中的滑动平均统计工具
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple


# ============================================================
# AverageMeter - 训练指标滑动平均统计
# ============================================================

class AverageMeter:
    """
    跟踪并计算指标的滑动平均值。
    常用于记录每个 epoch 内的平均 Loss。

    用法:
        meter = AverageMeter("train/loss")
        for batch in dataloader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=batch_size)
        print(f"平均 Loss: {meter.avg:.4f}")
        meter.reset()
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        更新统计量。

        Args:
            val: 当前批次的指标值
            n:   当前批次的样本数
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (val={self.val:.4f}, count={self.count})"


# ============================================================
# 动作误差计算
# ============================================================

def compute_action_error(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    reduction: str = "mean",
) -> Dict[str, float]:
    """
    计算预测动作与目标动作之间的误差。

    Args:
        pred_actions:   (B, chunk_size, action_dim) 预测动作序列
        target_actions: (B, chunk_size, action_dim) 目标动作序列
        reduction:      误差聚合方式 ("mean" 或 "none")

    Returns:
        字典包含:
          - "l1_error":  L1 误差 (MAE)
          - "l2_error":  L2 误差 (MSE)
          - "per_dim_l1": 每个动作维度的 L1 误差 (list)
    """
    with torch.no_grad():
        # L1 误差 (MAE)
        l1_error = torch.abs(pred_actions - target_actions)
        # L2 误差 (MSE)
        l2_error = (pred_actions - target_actions) ** 2

        if reduction == "mean":
            l1_mean = l1_error.mean().item()
            l2_mean = l2_error.mean().item()
            # 每个动作维度的平均 L1 误差: (action_dim,)
            per_dim_l1 = l1_error.mean(dim=[0, 1]).cpu().tolist()
        else:
            l1_mean = l1_error.mean().item()
            l2_mean = l2_error.mean().item()
            per_dim_l1 = l1_error.mean(dim=[0, 1]).cpu().tolist()

    return {
        "l1_error":   l1_mean,
        "l2_error":   l2_mean,
        "per_dim_l1": per_dim_l1,
    }


# ============================================================
# 成功率计算 (CALVIN 环境核心指标)
# ============================================================

def compute_success_rate(
    results: List[Dict],
    env_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    计算 CALVIN 环境中的任务成功率。

    CALVIN 评估协议:
      - 每个 episode 包含一系列连续子任务 (通常为 5 个)
      - 成功率 = 成功完成的子任务数 / 总子任务数
      - 还统计连续完成 1/2/3/4/5 个子任务的比例

    Args:
        results:  评估结果列表，每个元素为字典:
                  {
                    "episode_id": int,
                    "env":        str,       # 环境名称
                    "num_success": int,      # 成功完成的子任务数
                    "num_tasks":   int,      # 总子任务数
                    "task_results": list,    # 每个子任务的成功/失败 (bool)
                  }
        env_name: 若指定，则只统计该环境的结果

    Returns:
        字典包含:
          - "success_rate":     总体成功率 (成功子任务数 / 总子任务数)
          - "avg_tasks_done":   平均每个 episode 完成的子任务数
          - "chain_1":          至少完成 1 个连续子任务的 episode 比例
          - "chain_2":          至少完成 2 个连续子任务的 episode 比例
          - "chain_3":          至少完成 3 个连续子任务的 episode 比例
          - "chain_4":          至少完成 4 个连续子任务的 episode 比例
          - "chain_5":          完成全部 5 个连续子任务的 episode 比例
          - "num_episodes":     评估的 episode 总数
    """
    if env_name is not None:
        results = [r for r in results if r.get("env") == env_name]

    if len(results) == 0:
        return {
            "success_rate": 0.0,
            "avg_tasks_done": 0.0,
            "chain_1": 0.0, "chain_2": 0.0, "chain_3": 0.0,
            "chain_4": 0.0, "chain_5": 0.0,
            "num_episodes": 0,
        }

    total_tasks   = sum(r["num_tasks"]   for r in results)
    total_success = sum(r["num_success"] for r in results)
    num_episodes  = len(results)

    # 总体成功率
    success_rate = total_success / total_tasks if total_tasks > 0 else 0.0

    # 平均每个 episode 完成的子任务数
    avg_tasks_done = total_success / num_episodes

    # 连续完成 N 个子任务的 episode 比例 (CALVIN 标准指标)
    chain_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in results:
        task_results = r.get("task_results", [])
        # 计算从头开始的连续成功数
        consecutive = 0
        for success in task_results:
            if success:
                consecutive += 1
            else:
                break
        for n in range(1, 6):
            if consecutive >= n:
                chain_counts[n] += 1

    return {
        "success_rate":   success_rate,
        "avg_tasks_done": avg_tasks_done,
        "chain_1": chain_counts[1] / num_episodes,
        "chain_2": chain_counts[2] / num_episodes,
        "chain_3": chain_counts[3] / num_episodes,
        "chain_4": chain_counts[4] / num_episodes,
        "chain_5": chain_counts[5] / num_episodes,
        "num_episodes": num_episodes,
    }


def format_success_rate_table(
    results_by_env: Dict[str, Dict[str, float]],
) -> str:
    """
    将多环境成功率结果格式化为可读的表格字符串。

    Args:
        results_by_env: 字典，键为环境名，值为 compute_success_rate 的返回值

    Returns:
        格式化的表格字符串

    示例输出:
        ┌──────────┬──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
        │  环境    │ Success Rate │ Chain 1  │ Chain 2  │ Chain 3  │ Chain 4  │ Chain 5  │
        ├──────────┼──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
        │  Env A   │    72.3%     │  85.0%   │  68.0%   │  52.0%   │  38.0%   │  22.0%   │
        │  Env D   │    45.1%     │  60.0%   │  42.0%   │  28.0%   │  15.0%   │   8.0%   │
        └──────────┴──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
    """
    header = f"{'环境':^10} | {'Success Rate':^14} | {'Chain 1':^8} | {'Chain 2':^8} | {'Chain 3':^8} | {'Chain 4':^8} | {'Chain 5':^8}"
    sep    = "-" * len(header)
    rows   = [header, sep]

    for env_name, metrics in results_by_env.items():
        row = (
            f"{env_name:^10} | "
            f"{metrics['success_rate']*100:^13.1f}% | "
            f"{metrics['chain_1']*100:^7.1f}% | "
            f"{metrics['chain_2']*100:^7.1f}% | "
            f"{metrics['chain_3']*100:^7.1f}% | "
            f"{metrics['chain_4']*100:^7.1f}% | "
            f"{metrics['chain_5']*100:^7.1f}%"
        )
        rows.append(row)

    return "\n".join(rows)
