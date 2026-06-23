#!/bin/bash
# ============================================================
# ACT 训练脚本
# 用法: bash scripts/run_train.sh [single|joint] [可选参数]
# ============================================================

set -e  # 遇到错误立即退出

# --- 默认参数 ---
MODE=${1:-"single"}       # 训练模式: single (单环境) 或 joint (多环境联合)
DEVICE=${2:-"cuda:0"}     # 训练设备
RESUME=${3:-""}           # 断点恢复路径 (可选)

echo "=============================================="
echo "  ACT 跨环境泛化训练"
echo "  模式: ${MODE}"
echo "  设备: ${DEVICE}"
echo "=============================================="

# --- 选择配置文件 ---
if [ "${MODE}" = "single" ]; then
    CONFIG="configs/train_single_env.yaml"
    echo "  配置: 单环境 Baseline (环境 A)"
elif [ "${MODE}" = "joint" ]; then
    CONFIG="configs/train_joint_env.yaml"
    echo "  配置: 多环境联合训练 (环境 A+B+C)"
else
    echo "[Error] 未知模式: ${MODE}，请使用 'single' 或 'joint'"
    exit 1
fi

echo "  配置文件: ${CONFIG}"
echo "=============================================="

# --- 构建训练命令 ---
CMD="python train.py --config ${CONFIG} --device ${DEVICE}"

if [ -n "${RESUME}" ]; then
    CMD="${CMD} --resume ${RESUME}"
    echo "  断点恢复: ${RESUME}"
fi

# --- 执行训练 ---
echo ""
echo "执行命令: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "训练完成！"
