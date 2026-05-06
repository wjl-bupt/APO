#!/bin/bash

# V-MPO for Atari
python main.py --yaml ./conf/con_appo_run.yaml --algo vmpo --env_type atari --env_id BreakoutNoFrameskip-v4 \
    --num_envs 8 --learning_rate 2.5e-4 --ent_coef 0.01 \
    --update_epochs 4 --num_minibatches 4 --num_steps 128 \
    --total_timesteps 1000000 --vf_coef 0.5 --max_grad_norm 0.5 \
    --norm_adv --clip_vloss --decay_delta 0.995 \
    --entropy_coef 0.01 --gamma 0.99 \
    --vmpo_epsilon 0.1 --vmpo_epsilon_mean 0.1 --vmpo_epsilon_std 0.01 --vmpo_epsilon_value 0.05 \
    --vmpo_temperature_eta 1.0 --vmpo_temperature_alpha_mean 1.0 --vmpo_temperature_alpha_std 1.0 --vmpo_temperature_alpha_value 1.0