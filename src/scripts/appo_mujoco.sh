#!/bin/bash

echo "Mujoco Run Scripts"
exp="mujoco"
algo="appo"
seeds=(1 2 3 4 5)
update_epochs=(10)
envs=(
    # vcis 12
    "HumanoidStandup-v5"
    "Humanoid-v5"
    "Swimmer-v5"
    "HalfCheetah-v5"
    "Ant-v5"

    # vcis 13
    "Pusher-v5"
    "Hopper-v5"
    "Reacher-v5"
    "Walker2d-v5"
    "InvertedDoublePendulum-v5"
    "InvertedPendulum-v5"
)

# 循环遍历每个 seed 值，启动 main.py
# for env_id in "${envs[@]}"
# do
#     for seed in "${seeds[@]}"
#     do
#         for e in "${update_epochs[@]}"
#         do
#             echo "appo continous action space"
#             appo_yaml="src/conf/con_appo_run.yaml"
#             CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m src.main --seed $seed  --yaml $appo_yaml --env_id $env_id --env_type $exp --algo $algo
    
#         done
#         echo "Experiment with seed=$seed finished."
#     done
# done ps aux | grep "apo" | grep -v grep | awk '{print $2}' | xargs kill -9

for env_id in "${envs[@]}"
do
    for seed in "${seeds[@]}"
    do
        for e in "${update_epochs[@]}"
        do
            echo "appo continous action space"
            appo_yaml="src/conf/con_appo_run.yaml"
            CUDA_VISIBLE_DEVICES=1,2,3 python -m src.main \
                --seed $seed \
                --yaml $appo_yaml \
                --env_id $env_id \
                --env_type $exp \
                --algo $algo &
        done
        echo "Experiment with env=$env_id seed=$seed started."
    done
    wait
done


echo "All experiments finished."

