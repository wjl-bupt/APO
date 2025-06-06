# -*- encoding: utf-8 -*-
'''
@File       :dis_policy_value.py
@Description:
@Date       :2025/03/31 10:36:01
@Author     :junweiluo
@Version    :python
'''



import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Discrete_PolicyValue(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            # TODO(junweiluo): 离散环境测试梯度爆炸
            # nn.LayerNorm(512),
            nn.ReLU(),
            
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, state):
        return self.critic(self.encoder(state / 255.0)).squeeze(-1)

    def get_action_and_value(self, state, action=None):
        hidden = self.encoder(state / 255.0)
        distribution = Categorical(logits=self.actor(hidden))
        if action is None:
            action = distribution.sample()

        return action, distribution.log_prob(action), distribution.entropy(), self.critic(hidden).squeeze(-1), distribution.logits
