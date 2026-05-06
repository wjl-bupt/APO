# -*- encoding: utf-8 -*-
'''
@File       :vmpo_trainer.py
@Description:V-MPO (Virtual Continuous Multi-Step Off-Policy Correction) Algorithm Trainer
@Date       :2025/04/24
@Author     :Trae
@Version    :python
'''

import torch
import torch.nn as nn
import numpy as np
from ..ppo_family.base.base_trainer import BaseTrainer


class VMPOTeacherTrainer(BaseTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
        self.entropy_coef = args.entropy_coef if hasattr(args, 'entropy_coef') else 0.01
        self.epsilon = args.vmpo_epsilon if hasattr(args, 'vmpo_epsilon') else 0.1
        self.epsilon_mean = args.vmpo_epsilon_mean if hasattr(args, 'vmpo_epsilon_mean') else 0.1
        self.epsilon_std = args.vmpo_epsilon_std if hasattr(args, 'vmpo_epsilon_std') else 0.01
        self.epsilon_value = args.vmpo_epsilon_value if hasattr(args, 'vmpo_epsilon_value') else 0.05
        self.temperature_eta = args.vmpo_temperature_eta if hasattr(args, 'vmpo_temperature_eta') else 1.0
        self.temperature_alpha_mean = args.vmpo_temperature_alpha_mean if hasattr(args, 'vmpo_temperature_alpha_mean') else 1.0
        self.temperature_alpha_std = args.vmpo_temperature_alpha_std if hasattr(args, 'vmpo_temperature_alpha_std') else 1.0
        self.temperature_alpha_value = args.vmpo_temperature_alpha_value if hasattr(args, 'vmpo_temperature_alpha_value') else 1.0

    def update(self, data):
        """
        Perform one update step for V-MPO Teacher
        """
        b_obs, b_actions, b_returns, b_values, b_logprobs = data
        
        # Get new values and log probabilities
        _, new_logprob, entropy, new_value, new_mean_std = self.agent.get_action_and_value(b_obs, b_actions)
        
        # Calculate advantages (returns - values)
        advantages = b_returns - b_values
        
        # Normalize advantages if specified
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # V-MPO specific losses
        # Compute importance weights
        log_ratio = new_logprob - b_logprobs
        ratio = torch.exp(log_ratio)
        
        # Policy improvement term
        policy_improvement_loss = -(advantages.detach() * ratio).mean()
        
        # KL divergence regularization terms for policy
        kl_divergence_loss = torch.mean((ratio - 1) - log_ratio)
        
        # Value loss
        value_loss = self.compute_value_loss(b_returns, b_values, new_value)
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # V-MPO teacher loss with temperature parameters
        total_loss = (
            policy_improvement_loss +
            self.temperature_alpha_mean * kl_divergence_loss +
            self.temperature_alpha_value * value_loss -
            self.entropy_coef * entropy_loss
        )
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Log metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfrac = ((ratio - 1.0).abs() > 0.02).float().mean()
            
            mini_dict_ = self.log_dict_(
                losses_value_loss=value_loss.item(),
                losses_policy_loss=policy_improvement_loss.item(),
                losses_entropy=entropy_loss.item(),
                losses_total_loss=total_loss.item(),
                losses_approx_kl=approx_kl.item(),
                losses_clipfrac=clipfrac.item(),
                train_grad_norm=grad_norm.item()
            )
        
        return mini_dict_