#!/bin/bash

echo "Mujoco Run Scripts"
exp="mujoco"
algo="spo"
seeds=(1 2 3 4 5)
update_epochs=(10)
envs=(
    # vcis 12
    # "HumanoidStandup-v5"
    # "Humanoid-v5"
    # "Swimmer-v5"

    # vcis 7
    # "HalfCheetah-v5"
    "Ant-v5"

    # vcis 13
    # "Hopper-v5"
    # "Reacher-v5"
    # "Walker2d-v5"
    # "InvertedDoublePendulum-v5"
    # "InvertedPendulum-v5"
)


# set root dir, it will return "src/" abs path
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
CONFIG_PATH="$PROJECT_ROOT/conf/con_ppo2_run.yaml"

# check config file
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi


# diff seeds & diff envs
for env_id in "${envs[@]}"
do
    for seed in "${seeds[@]}"
    do
        for e in "${update_epochs[@]}"
        do
            echo "ppo-clip continous action space"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main --seed $seed  --yaml $CONFIG_PATH --env_id $env_id --env_type $exp   --algo $algo     
        done
        echo "Experiment with env=$env_id seed=$seed finished."
    done
done

