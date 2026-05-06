#!/bin/bash

# TRPO for Mujoco
python main.py --yaml ./conf/con_appo_run.yaml --algo trpo --env_type mujoco --env_id HalfCheetah-v4 \
    --num_envs 1 --learning_rate 3e-4 --ent_coef 0.0 \
    --update_epochs 1 --num_minibatches 1 --num_steps 2048 \
    --total_timesteps 1000000 --vf_coef 0.25 --max_grad_norm 0.5 \
    --norm_adv --clip_vloss --decay_delta 0.995 \
    --max_kl 0.01 --cg_damping 0.1 --line_search_coef 0.8 --max_backtracks 10