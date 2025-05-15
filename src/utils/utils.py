# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/03/25 16:58:34
@Author  :   junewluo 
'''

import torch
import gymnasium as gym
import numpy as np

def compute_advantages(args, agent, rewards, values, next_obs, next_done, dones):
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(args.device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    return returns, advantages

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# # 预处理函数列表
# env_pre_funcs = [
#     lambda env: gym.wrappers.FlattenObservation(env),  # 展平观测
#     lambda env: gym.wrappers.RecordEpisodeStatistics(env),  # 记录统计信息
#     lambda env: gym.wrappers.ClipAction(env),  # 裁剪动作
#     lambda env: gym.wrappers.NormalizeObservation(env),  # 归一化观测
#     lambda env: gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10)),  # 裁剪观测
#     lambda env: gym.wrappers.NormalizeReward(env, gamma=gamma),  # 归一化奖励
#     lambda env: gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))  # 裁剪奖励
# ]



def compute_kld(mu_1, sigma_1, mu_2, sigma_2):
    """ only suitable for continous action space """
    return torch.log(sigma_2 / sigma_1) + ((mu_1 - mu_2) ** 2 + (sigma_1 ** 2 - sigma_2 ** 2)) / (2 * sigma_2 ** 2)