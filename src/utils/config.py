# -*- encoding: utf-8 -*-
'''
@File       :config.py
@Description:configuration experiment.
@Date       :2025/03/27 10:00:22
@Author     :junweiluo
@Version    :python
'''
import argparse
import os
import torch
import time
import yaml
from .mylogging import getLogger
from prettytable import PrettyTable


def add_fixed_arguments(parser, args = None):
    parser.add_argument('--env_type', type=str, default=args.env_type if args != None else None, help="Type of Env", choices=["atari", "mujoco"])
    parser.add_argument('--env_id', type=str, default=args.env_id if args != None else None, help="The id of the environment")
    parser.add_argument('--yaml', type=str, default=args.yaml if args != None else None, help="configuration file to launch exp through toml file!")
    parser.add_argument('--seed', type=int, default=args.seed if args != None else None, help="The id of the environment")
    parser.add_argument('--algo', type=str, default=args.algo if args != None else None, help="Which algorithm to test", choices=["appo","ppo-clip", "ppo-penalty", "ppo2-kl","ppo2-ent", "spo"])

def get_conf():
    parser = argparse.ArgumentParser(description="Anchor PPO Exeperiment")
    # toml config
    add_fixed_arguments(parser)
    args, remaining_argv = parser.parse_known_args()

    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"{args.yaml}")
    with open(args.yaml, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Re-create argument parser to handle config values
    parser = argparse.ArgumentParser(description="Anchor PPO Experiment with Config Overwrite")
    add_fixed_arguments(parser, args)
    for BigClass, Dict_ in yaml_config.items():
        for param, value in Dict_.items():
            parser.add_argument(f"--{param}", type=type(value), default=value)

    # Parse remaining arguments, allowing command-line overrides
    args = parser.parse_args(remaining_argv)
    # if args.env_type == "atari":
    args.exp_name = f'{args.env_id}_{args.algo}_update{args.update_epochs}_clipcoef{args.clip_coef}_sample{args.sample_action_num}_{args.decay_delta}'
    # else:
    # args.exp_name = f'{args.env_id}_seed{args.seed}_update{args.update_epochs}_clipcoef{args.clip_coef}_sample{args.sample_action_num}'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = getLogger(f"{args.env_id}_{int(time.time())}", "colored")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Print configuration in a table
    config_table = PrettyTable()
    config_table.field_names = ["Parameter", "Value"]
    for arg in vars(args):
        config_table.add_row([arg, getattr(args, arg)])
    args.logger.info(f"\n🔍 Configuration Table\n{config_table}")
    
    return args
