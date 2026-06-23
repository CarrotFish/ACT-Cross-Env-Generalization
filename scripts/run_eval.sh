#!/bin/bash
# ============================================================
# ACT 评估脚本
# 用法: bash scripts/run_eval.sh [offline|online] [checkpoint] [eval_envs...]
# ============================================================

set -e

# --- 默认参数 ---
MODE=${1:-"offline"}                                          # 评估模式
CHECKPOINT=${2:-"checkpoints/joint_env_ABC/best_model.pth"}  # 模型权重路径
EVAL_ENVS=${3:-"A B C D"}                                     # 评估环境列表
CONFIG=${4:-"configs/train_joint_env.yaml"}                   # 配置文件
DEVICE=${5:-"cuda:0"}                                         # 评估设备
NUM_EPISODES=${6:-100}                                        # 在线评估 episode 数

echo "=============================================="
echo "  ACT 跨环境泛化评估"
echo "  模式:     ${MODE}"
echo "  权重:     ${CHECKPOINT}"
echo "  环境:     ${EVAL_ENVS}"
echo "  配置:     ${CONFIG}"
echo "  设备:     ${DEVICE}"
echo "=============================================="

# --- 检查权重文件是否存在 ---
if [ ! -f "${CHECKPOINT}" ]; then
    echo "[Error] 权重文件不存在: ${CHECKPOINT}"
    echo "  请先运行训练: bash scripts/run_train.sh joint"
    exit 1
fi

# --- 构建评估命令 ---
CMD="python eval.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --eval_envs ${EVAL_ENVS} \
    --mode ${MODE} \
    --device ${DEVICE} \
    --output_dir ./results"

if [ "${MODE}" = "online" ]; then
    CMD="${CMD} --num_episodes ${NUM_EPISODES}"
fi

# --- 执行评估 ---
echo ""
echo "执行命令: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "评估完成！结果已保存至 ./results/"

# ============================================================
# 快捷命令示例:
#
# 1. 离线评估 (快速验证，无需仿真环境)
#    bash scripts/run_eval.sh offline checkpoints/joint_env_ABC/best_model.pth "A B C D"
#
# 2. Zero-shot 在线评估 (环境 D)
#    bash scripts/run_eval.sh online checkpoints/joint_env_ABC/best_model.pth "D" configs/train_joint_env.yaml cuda:0 100
#
# 3. 对比实验: 单环境模型在环境 D 上的 Zero-shot 性能
#    bash scripts/run_eval.sh online checkpoints/single_env_A/best_model.pth "D" configs/train_single_env.yaml cuda:0 100
# ============================================================
