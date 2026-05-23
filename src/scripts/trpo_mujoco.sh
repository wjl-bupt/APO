#!/bin/bash

# ==========================================
# TRPO Mujoco-v5 Batch Runner
# 每个环境并行运行 5 个 seed
# 当前环境全部结束后，再运行下一个环境
# ==========================================

# Mujoco v5 environments
ENVS=(
    "HalfCheetah-v5"
    "Hopper-v5"
    "Walker2d-v5"
    "Ant-v5"
    "Humanoid-v5"
    "Swimmer-v5"
    "Reacher-v5"
    "Pusher-v5"
    "InvertedPendulum-v5"
    "InvertedDoublePendulum-v5"
)

# Seeds
SEEDS=(0 1 2 3 4)

# 日志目录
LOG_DIR=logs_trpo_mujoco_v5
mkdir -p ${LOG_DIR}

# Python入口
PYTHON_BIN=python

# 配置文件
CONFIG=/root/APO/src/conf/con_trpo_run.yaml

# 遍历环境
for ENV_ID in "${ENVS[@]}"; do

    echo "========================================="
    echo "Starting environment: ${ENV_ID}"
    echo "========================================="

    # 当前环境启动5个seed（并行）
    for SEED in "${SEEDS[@]}"; do

        echo "Launching ${ENV_ID} | seed=${SEED}"

        CUDA_VISIBLE_DEVICES="1" python src/main.py \
            --yaml ${CONFIG} \
            --algo trpo \
            --env_type mujoco \
            --env_id ${ENV_ID} \
            --num_envs 8 \
            --seed ${SEED} \
            --max_kl 0.01 \
            --cg_damping 0.1 \
            --line_search_coef 0.8 \
            --max_backtracks 10 \
            > ${LOG_DIR}/${ENV_ID}_seed${SEED}.log 2>&1 &

    done

    # 等待当前环境的5个seed全部完成
    wait

    echo "Finished environment: ${ENV_ID}"
    echo ""

done

echo "========================================="
echo "All TRPO Mujoco-v5 experiments finished!"
echo "========================================="