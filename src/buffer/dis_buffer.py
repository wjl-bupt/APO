# -*- encoding: utf-8 -*-
'''
@File       :atari_buffer.py
@Description:
@Date       :2025/03/26 18:47:42
@Author     :junweiluo
@Version    :python
'''

import torch
from .base_buffer import BaseBuffer

class AtariBuffer(BaseBuffer):
    def __init__(self, args):
        super().__init__(args)
        self.obs = torch.zeros((args.num_steps, args.num_envs,) + args.single_observation_space.shape).to(args.device)
        self.actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(args.device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(args.device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(args.device)
        self.log_probs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(args.device)
        self.values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(args.device)
        self.total_logits = torch.zeros((args.num_steps, args.num_envs, args.discrete_action_space_n), dtype=torch.float32).to(args.device)


    def push(self, obs, action, reward, done, log_prob, value, total_logits, step):
        self.obs[step] = obs
        self.actions[step] = action
        self.rewards[step] = reward
        self.dones[step] = done
        self.log_probs[step] = log_prob
        self.values[step] = value
        # junweiluo
        self.total_logits[step] = total_logits

    def pop(self):
        return (
            self.obs,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.values,
            self.total_logits,
        )