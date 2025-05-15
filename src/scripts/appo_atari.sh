#!/bin/bash

exp="atari"
seeds=(1 2 3)
update_epochs=(4)
algo="appo"
envs=(
    "ALE/Assault-v5"
    # "ALE/Alien-v5"
    # "ALE/Amidar-v5"
    # "ALE/Asterix-v5"
    # "ALE/Asteroids-v5"
    # "ALE/Atlantis-v5"
    # "ALE/BankHeist-v5"
    # "ALE/BattleZone-v5"
    # "ALE/BeamRider-v5"
    # "ALE/Berzerk-v5"
    # "ALE/Bowling-v5"
    # "ALE/Boxing-v5"
    # "ALE/Breakout-v5"
    # "ALE/Centipede-v5"
    # "ALE/ChopperCommand-v5"
    # "ALE/CrazyClimber-v5"
    # "ALE/DemonAttack-v5"
    # "ALE/DoubleDunk-v5"
    # "ALE/Enduro-v5"
    # "ALE/FishingDerby-v5"
    # "ALE/Freeway-v5"
    # "ALE/Frostbite-v5"
    # "ALE/Gopher-v5"
    # "GravitarNoFrameskip-v4"
    # "ALE/Hero-v5"
    # "ALE/IceHockey-v5"
    # "ALE/Jamesbond-v5"
    # "ALE/Kangaroo-v5"
    # "ALE/Krull-v5"
    # "ALE/KungFuMaster-v5"
    # "ALE/MontezumaRevenge-v5"
    # "ALE/MsPacman-v5"
    # "ALE/NameThisGame-v5"
    # "ALE/Phoenix-v5"
    # "ALE/Pitfall-v5"
    # "ALE/Pong-v5"
    # "ALE/PrivateEye-v5"
    # "ALE/Qbert-v5"
    # "ALE/Riverraid-v5"
    # "ALE/RoadRunner-v5"
    # "ALE/Robotank-v5"
    # "ALE/Seaquest-v5"
    # "ALE/SpaceInvaders-v5"
    # "ALE/StarGunner-v5"
    # "ALE/Tennis-v5"
    # "ALE/TimePilot-v5"
    # "ALE/Tutankham-v5"
    # "ALE/UpNDown-v5"
    # "ALE/Venture-v5"
    # "ALE/VideoPinball-v5"
    # "ALE/WizardOfWor-v5"
    # "ALE/YarsRevenge-v5"
    # "ALE/Zaxxon-v5"

)

# set root dir, it will return "src/" abs path
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
CONFIG_PATH="$PROJECT_ROOT/conf/dis_exp_run.yaml"

# check config file
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# 循环遍历每个 seed 值，启动 main.py
for env_id in "${envs[@]}"
do
    for seed in "${seeds[@]}"
    do
        for e in "${update_epochs[@]}"
        do
            CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main --seed $seed  --yaml $CONFIG_PATH --env_id $env_id --env_type $exp --algo $algo

            # echo "ppo-clip continous action space"
            # ppoclip_yaml="/home/wangchenxu/ppobased-mujuco/src/conf/con_ppoclip_run.yaml"
            # /home/wangchenxu/anaconda3/envs/mujoco_v4/bin/python ./src/algos/mujoco/run.py --seed $seed  --yaml $appo_yaml --env_id $env_id --env_type $exp         
        done
        echo "Experiment with env_id=$env_id seed=$seed finished."
    done
done