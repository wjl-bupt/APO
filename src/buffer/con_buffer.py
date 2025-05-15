# -*- encoding: utf-8 -*-
'''
@File       :mujoco_buffer.py
@Description:
@Date       :2025/03/26 16:30:34
@Author     :junweiluo
@Version    :python
'''
import torch
from .base_buffer import BaseBuffer

class MujocoBuffer(BaseBuffer):
    def __init__(self, args):
        super().__init__(args)
        self.obs = torch.zeros((args.num_steps, args.num_envs) + args.single_observation_space.shape).to(args.device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + (args.sample_action_num, ) + args.single_action_space.shape).to(args.device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)+ (args.sample_action_num, )).to(args.device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        self.means = torch.zeros((args.num_steps, args.num_envs) + args.single_action_space.shape).to(args.device)
        self.stds = torch.zeros((args.num_steps, args.num_envs) + args.single_action_space.shape).to(args.device)
    
    def push(self, data, step):
        obs, actions, logprobs, rewards, dones, values, means, stds = data
        self.obs[step] = obs
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        self.means[step] = means
        self.stds[step] = stds
    
    def pop(self):
        
        return (
            self.obs,
            self.actions,
            self.logprobs, 
            self.rewards,
            self.dones,
            self.values,
            self.means,
            self.stds,
        )
        