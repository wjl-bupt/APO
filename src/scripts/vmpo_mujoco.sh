#!/bin/bash

# V-MPO for Mujoco
python main.py --yaml ./conf/con_appo_run.yaml --algo vmpo --env_type mujoco --env_id HalfCheetah-v4 \
    --num_envs 1 --learning_rate 3e-4 --ent_coef 0.0 \
    --update_epochs 10 --num_minibatches 32 --num_steps 2048 \
    --total_timesteps 1000000 --vf_coef 0.25 --max_grad_norm 0.5 \
    --norm_adv --clip_vloss --decay_delta 0.995 \
    --entropy_coef 0.01 --gamma 0.99 \
    --vmpo_epsilon 0.1 --vmpo_epsilon_mean 0.1 --vmpo_epsilon_std 0.01 --vmpo_epsilon_value 0.05 \
    --vmpo_temperature_eta 1.0 --vmpo_temperature_alpha_mean 1.0 --vmpo_temperature_alpha_std 1.0 --vmpo_temperature_alpha_value 1.0