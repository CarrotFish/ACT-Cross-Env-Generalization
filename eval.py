"""
ACT 跨环境泛化评估脚本

支持:
  - 同分布评估 (In-distribution): 在训练环境上测试
  - Zero-shot 跨环境评估:         在未见过的环境 D 上测试
  - 批量对比实验:                  同时评估多个模型权重

用法:
  # 在环境 D 上进行评估
  python eval.py --config configs/train_joint_env.yaml --checkpoint checkpoints/joint_env_ABC/best_model.pth --eval_envs D

  # 在所有环境上评估 (含 Zero-shot)
  python eval.py --config configs/train_joint_env.yaml --checkpoint checkpoints/best_model.pth --eval_envs A B C D

  # 对比单环境与多环境模型
  python eval.py --config configs/base_act.yaml --checkpoint checkpoints/single_env_A/best_model.pth --eval_envs D
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from datasets import build_dataloader
from models   import build_model
from utils    import compute_success_rate, compute_action_error, AverageMeter
from utils.metrics import format_success_rate_table
from train    import load_config


# ============================================================
# 离线评估 (基于数据集的 Action L1 Loss 评估)
# ============================================================

def _compute_offline_success_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    threshold_factor: float = 0.5,
) -> Dict[str, float]:
    """
    基于动作误差阈值计算离线"模拟成功率"。
    
    CALVIN 动作格式 (通常为 7 维):
      - [0:3]: 末端执行器相对位置变化 (delta x, y, z)
      - [3:6]: 末端执行器相对欧拉角变化 (delta roll, pitch, yaw)
      - [6]:   夹钳开合状态 (0=打开, 1=闭合)
    
    阈值设定 (基于 CALVIN 常见标准):
      - 位置误差 < 5cm (L1)
      - 角度误差 < 5° (L1) 
      - 夹钳状态完全匹配
    
    Args:
        pred_actions:   (B, chunk_size, action_dim) 预测动作
        target_actions: (B, chunk_size, action_dim) 目标动作
        threshold_factor: 阈值因子，1.0 表示严格阈值，0.5 表示宽松阈值
        
    Returns:
        字典包含模拟成功率指标
    """
    with torch.no_grad():
        # 计算每个样本每个时间步的 L1 误差
        l1_per_step = torch.abs(pred_actions - target_actions)  # (B, chunk_size, action_dim)
        
        # 阈值设定:
        # CALVIN 动作格式为相对位移 (delta x, y, z) + 相对欧拉角 (delta roll, pitch, yaw) + 夹钳状态
        # 使用固定阈值进行判断
        pos_threshold = 0.1  # 位置误差 < 10cm
        orn_threshold = 0.15  # 角度误差 < ~8.6度 (弧度)
        gripper_threshold = 0.5  # 夹钳状态必须匹配
        
        # 宽松阈值 (用于参考)
        pos_threshold_loose = 0.2  # 位置误差 < 20cm
        orn_threshold_loose = 0.30  # 角度误差 < ~17.2度 (弧度)
        gripper_threshold_loose = 1.0
        
        B, chunk_size, action_dim = pred_actions.shape
        
        # 计算每个样本的"成功"次数 (所有时间步都满足阈值)
        # 对于每个时间步，检查所有维度的误差
        pos_error = l1_per_step[:, :, :3].mean(dim=-1)  # (B, chunk_size) - 平均位置误差
        orn_error = l1_per_step[:, :, 3:6].mean(dim=-1)  # (B, chunk_size) - 平均角度误差
        gripper_error = l1_per_step[:, :, 6:]  # (B, chunk_size, 1) - 夹钳误差
        
        # 严格阈值下的成功判断
        pos_success = pos_error < pos_threshold
        orn_success = orn_error < orn_threshold
        gripper_success = gripper_error.squeeze(-1) < gripper_threshold
        
        # 每个时间步是否"成功"
        step_success = pos_success & orn_success & gripper_success  # (B, chunk_size)
        
        # 整个序列是否"成功" (所有时间步都成功)
        seq_success = step_success.all(dim=1)  # (B,)
        
        # 宽松阈值下的成功判断
        pos_success_loose = pos_error < pos_threshold_loose
        orn_success_loose = orn_error < orn_threshold_loose
        gripper_success_loose = gripper_error.squeeze(-1) < gripper_threshold_loose
        
        step_success_loose = pos_success_loose & orn_success_loose & gripper_success_loose
        seq_success_loose = step_success_loose.all(dim=1)
        
        # 统计结果
        strict_success_rate = seq_success.float().mean().item()
        loose_success_rate = seq_success_loose.float().mean().item()
        
        # 每个动作维度的平均 L1 误差
        per_dim_l1 = l1_per_step.mean(dim=[0, 1]).cpu().tolist()
        
        # 位置误差统计
        pos_l1 = l1_per_step[:, :, :3].mean(dim=[0, 1]).cpu().tolist()
        orn_l1 = l1_per_step[:, :, 3:6].mean(dim=[0, 1]).cpu().tolist()
        gripper_l1 = l1_per_step[:, :, 6:].mean(dim=[0, 1]).cpu().tolist()
        
        # 每个时间步的平均成功率和位置/角度误差
        avg_step_success_rate = step_success.float().mean().item()
        avg_pos_error = pos_error.mean(dim=1).mean().item()
        avg_orn_error = orn_error.mean(dim=1).mean().item()
        
        # 动作误差的分位数 (用于更详细的分析)
        l1_flat = l1_per_step.flatten().cpu().tolist()
        l1_median = np.median(l1_flat)
        l1_p90 = np.percentile(l1_flat, 90)
        l1_p95 = np.percentile(l1_flat, 95)
        l1_p99 = np.percentile(l1_flat, 99)

    return {
        # 严格阈值下的模拟成功率
        "simulated_success_rate_strict": strict_success_rate,
        # 宽松阈值下的模拟成功率
        "simulated_success_rate_loose": loose_success_rate,
        # 每个时间步的平均成功率
        "avg_step_success_rate": avg_step_success_rate,
        # 平均位置误差 (每维)
        "avg_pos_error_per_dim": pos_l1,
        # 平均角度误差 (每维，弧度)
        "avg_orn_error_per_dim": orn_l1,
        # 平均夹钳误差
        "avg_gripper_error": gripper_l1,
        # 总平均位置误差
        "avg_pos_error": avg_pos_error,
        # 总平均角度误差 (弧度)
        "avg_orn_error": avg_orn_error,
        # 动作误差统计
        "l1_median": l1_median,
        "l1_p90": l1_p90,
        "l1_p95": l1_p95,
        "l1_p99": l1_p99,
        # 每维度 L1 误差
        "per_dim_l1": per_dim_l1,
    }


@torch.no_grad()
def evaluate_offline(
    model,
    dataloader,
    device: torch.device,
    env_name: str,
    cfg: dict,
) -> Dict:
    """
    离线评估模式：在数据集上计算 Action L1 Loss 和动作误差。
    不需要运行真实的 CALVIN 仿真环境。
    
    新增指标:
      - simulated_success_rate_strict: 基于严格动作误差阈值的模拟成功率
      - simulated_success_rate_loose: 基于宽松动作误差阈值的模拟成功率
      - avg_pos_error: 平均位置误差
      - avg_orn_error: 平均角度误差
      - 动作误差分位数统计

    Args:
        model:      ACTWrapper 模型
        dataloader: 评估数据加载器
        device:     评估设备
        env_name:   环境名称
        cfg:        配置字典

    Returns:
        评估指标字典
    """
    model.eval()
    use_amp = cfg.get("hardware", {}).get("use_amp", False)

    from torch.cuda.amp import autocast

    l1_meter = AverageMeter(f"{env_name}/l1_error")
    l2_meter = AverageMeter(f"{env_name}/l2_error")

    all_per_dim_l1 = []
    all_sim_success = []  # 存储每个样本的模拟成功结果
    
    # 收集所有样本的动作误差用于分位数统计
    all_l1_errors = []

    for batch in dataloader:
        image   = batch["image"].to(device, non_blocking=True)
        state   = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        B = image.shape[0]

        with autocast(enabled=use_amp):
            # 推理模式: 不传入 actions (z ~ N(0,I))
            outputs = model(image, state, actions=None)
            pred_actions = outputs["pred_actions"]

        # 计算动作误差
        errors = compute_action_error(pred_actions, actions)

        l1_meter.update(errors["l1_error"], n=B)
        l2_meter.update(errors["l2_error"], n=B)
        all_per_dim_l1.append(errors["per_dim_l1"])
        
        # 收集 L1 误差用于分位数统计
        l1_per_sample = torch.abs(pred_actions - actions).flatten().cpu().tolist()
        all_l1_errors.extend(l1_per_sample)
        
        # 计算每个样本的模拟成功结果
        sim_metrics = _compute_offline_success_metrics(pred_actions, actions)
        all_sim_success.append({
            "strict": sim_metrics["simulated_success_rate_strict"] * B,
            "loose": sim_metrics["simulated_success_rate_loose"] * B,
        })

    # 计算每个动作维度的平均 L1 误差
    per_dim_l1_avg = np.mean(all_per_dim_l1, axis=0).tolist()
    
    # 计算总体模拟成功率
    total_strict_success = sum(s["strict"] for s in all_sim_success)
    total_loose_success = sum(s["loose"] for s in all_sim_success)
    
    # 重新计算总样本数
    total_samples = 0
    for batch in dataloader:
        total_samples += batch["image"].shape[0]
    
    overall_strict_success_rate = total_strict_success / total_samples if total_samples > 0 else 0.0
    overall_loose_success_rate = total_loose_success / total_samples if total_samples > 0 else 0.0
    
    # 计算动作误差分位数
    all_l1_errors_np = np.array(all_l1_errors)
    l1_median = float(np.median(all_l1_errors_np))
    l1_p90 = float(np.percentile(all_l1_errors_np, 90))
    l1_p95 = float(np.percentile(all_l1_errors_np, 95))
    l1_p99 = float(np.percentile(all_l1_errors_np, 99))

    return {
        "env":                              env_name,
        "l1_error":                         l1_meter.avg,
        "l2_error":                         l2_meter.avg,
        "per_dim_l1":                       per_dim_l1_avg,
        "num_samples":                      l1_meter.count,
        # 新增：模拟成功率指标
        "simulated_success_rate_strict":    overall_strict_success_rate,
        "simulated_success_rate_loose":     overall_loose_success_rate,
        # 新增：动作误差分位数
        "l1_median":                        l1_median,
        "l1_p90":                           l1_p90,
        "l1_p95":                           l1_p95,
        "l1_p99":                           l1_p99,
        # 新增：每维度误差详情
        "avg_pos_error_per_dim":            sim_metrics.get("avg_pos_error_per_dim", []),
        "avg_orn_error_per_dim":            sim_metrics.get("avg_orn_error_per_dim", []),
        "avg_gripper_error":                sim_metrics.get("avg_gripper_error", []),
    }


# ============================================================
# 主评估函数
# ============================================================

def evaluate(
    cfg: dict,
    checkpoint_path: str,
    eval_envs: List[str],
    output_dir: str = "./results",
):
    """
    主评估函数。

    Args:
        cfg:             完整配置字典
        checkpoint_path: 模型权重路径
        eval_envs:       要评估的环境列表
        output_dir:      结果保存目录
    """
    hw_cfg = cfg.get("hardware", {})
    device = torch.device(hw_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"[Device] 使用设备: {device}")

    # --- 加载模型 ---
    model = build_model(cfg).to(device)
    model.load_checkpoint(checkpoint_path, device=str(device))
    model.eval()

    print(f"\n[Eval] 评估环境: {eval_envs}")
    print(f"[Eval] 权重文件: {checkpoint_path}\n")

    # --- 执行评估 ---
    all_results = {}

    for env_name in eval_envs:
        print(f"{'─'*50}")
        print(f"评估环境: {env_name}")
        print(f"{'─'*50}")

        # 离线评估: 使用数据集计算动作误差
        eval_loader = build_dataloader(
            cfg,
            env_names=[env_name],
            split="validation",
            is_train=False,
        )
        metrics = evaluate_offline(model, eval_loader, device, env_name, cfg)
        print(
            f"  L1 Error: {metrics['l1_error']:.4f} | "
            f"L2 Error: {metrics['l2_error']:.4f} | "
            f"Samples: {metrics['num_samples']}"
        )
        print(
            f"  Simulated Success Rate (Strict): {metrics.get('simulated_success_rate_strict', 0)*100:.1f}% | "
            f"(Loose): {metrics.get('simulated_success_rate_loose', 0)*100:.1f}%"
        )
        print(
            f"  Action Error Percentiles - Median: {metrics.get('l1_median', 0):.4f}, "
            f"P90: {metrics.get('l1_p90', 0):.4f}, "
            f"P95: {metrics.get('l1_p95', 0):.4f}, "
            f"P99: {metrics.get('l1_p99', 0):.4f}"
        )

        all_results[env_name] = metrics

    # --- 保存结果 ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(checkpoint_path).stem
    result_file = os.path.join(output_dir, f"eval_{ckpt_name}_{timestamp}.json")

    save_data = {
        "checkpoint":  checkpoint_path,
        "eval_envs":   eval_envs,
        "timestamp":   timestamp,
        "results":     all_results,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"[Eval] 评估结果已保存至: {result_file}")

    return all_results


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ACT 跨环境泛化评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 在环境 D 上进行评估
  python eval.py --config configs/train_joint_env.yaml \\
                 --checkpoint checkpoints/joint_env_ABC/best_model.pth \\
                 --eval_envs D

  # 在所有环境上评估 (含 Zero-shot)
  python eval.py --config configs/train_joint_env.yaml \\
                 --checkpoint checkpoints/best_model.pth \\
                 --eval_envs A B C D

  # 对比单环境与多环境模型
  python eval.py --config configs/base_act.yaml \\
                 --checkpoint checkpoints/single_env_A/best_model.pth \\
                 --eval_envs D
        """
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型权重文件路径 (.pth)"
    )
    parser.add_argument(
        "--eval_envs", type=str, nargs="+", default=["D"],
        help="要评估的环境列表 (e.g., --eval_envs A B C D)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="评估结果保存目录"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="覆盖配置中的设备"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行参数覆盖
    if args.device is not None:
        cfg.setdefault("hardware", {})["device"] = args.device

    # 执行评估
    evaluate(
        cfg             = cfg,
        checkpoint_path = args.checkpoint,
        eval_envs       = args.eval_envs,
        output_dir      = args.output_dir,
    )