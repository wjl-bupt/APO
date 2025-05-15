#!/bin/bash

# 运行 ppo_mujoco.sh，并将日志写入 mujoco_ppo.log
nohup bash ./src/scripts/ppo_mujoco.sh > ./src/scripts/mujoco_ppo.log 2>&1 &
PPO_PID=$!  # 获取 ppo_mujoco.sh 的进程 ID

# 运行 appo_mujoco.sh，并将日志写入 mujoco_appo.log
nohup bash ./src/scripts/appo_mujoco.sh > ./src/scripts/mujoco_appo.log 2>&1 &
APPO_PID=$!  # 获取 appo_mujoco.sh 的进程 ID

echo "appo pid: $APPO_PID; ppo pid: $PPO_PID"
# 等待两个脚本同时完成
wait $PPO_PID
wait $APPO_PID

echo "All processes completed!"
