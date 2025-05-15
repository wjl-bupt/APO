# -*- encoding: utf-8 -*-
'''
@File       :con_network.py
@Description:
@Date       :2025/03/31 10:24:23
@Author     :junweiluo
@Version    :python
'''

# -*- encoding: utf-8 -*-
'''
@File    :   agent.py
@Time    :   2025/03/25 22:17:24
@Author  :   junewluo 
'''

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
from utils import layer_init

# policy & value network
class Continous_PolicyValue(nn.Module):
    def __init__(self, envs, sample_action_num = 1):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.sample_action_num = sample_action_num 
    
    # sample action function
    def sample_action(self, probs):
        actions = []
        for _ in range(self.sample_action_num):
            # there is no need use tanh() to scale action range.
            i_action = probs.sample()
            actions.append(i_action)
        actions = torch.stack(actions, dim = 1)
        log_probs = self.get_logprobs(actions, probs)

        return actions, log_probs

    def get_logprobs(self, actions, probs):
        """ actions shape is [num_envs, self.sample_action_num, action_dim] """
        log_probs = []
        for i in range(self.sample_action_num):
            log_probs.append(probs.log_prob(actions[:,i,:]))
        
        return torch.stack(log_probs, dim = 1).sum(2)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            # action shape is (num_envs, sample_action_num, action_dim)
            action, log_probs = self.sample_action(probs)
            return action, log_probs, probs.entropy().sum(1), self.critic(x), probs
        
        # ppo更新时计算新的log_probs
        log_probs = self.get_logprobs(actions = action, probs = probs)
        return action, log_probs, probs.entropy().sum(1), self.critic(x), probs