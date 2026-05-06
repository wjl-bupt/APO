# -*- encoding: utf-8 -*-
'''
@File       :a2c_trainer.py
@Description:A2C (Advantage Actor-Critic) Algorithm Trainer
@Date       :2025/04/24
@Author     :Trae
@Version    :python
'''

import torch
import torch.nn as nn
import numpy as np
from ..ppo_family.base.base_trainer import BaseTrainer


class A2CTrainer(BaseTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
        self.entropy_coef = args.entropy_coef if hasattr(args, 'entropy_coef') else 0.01
        self.gamma = args.gamma if hasattr(args, 'gamma') else 0.99
        self.value_loss_coef = args.value_loss_coef if hasattr(args, 'value_loss_coef') else 0.25

    def update(self, data):
        """
        Perform one update step for A2C
        """
        b_obs, b_actions, b_returns, b_values, b_logprobs = data
        
        # Get new values and log probabilities
        _, new_logprob, entropy, new_value, _ = self.agent.get_action_and_value(b_obs, b_actions)
        
        # Calculate advantages (returns - values)
        advantages = b_returns - b_values
        
        # Normalize advantages if specified
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate policy loss (negative of expected advantage)
        ratio = torch.exp(new_logprob - b_logprobs)
        pg_loss = -(advantages.detach() * ratio).mean()
        
        # Calculate value loss
        value_loss = self.compute_value_loss(b_returns, b_values, new_value)
        
        # Calculate entropy loss
        entropy_loss = entropy.mean()
        
        # Total loss
        total_loss = pg_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Log metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (new_logprob - b_logprobs)).mean()
            clipfrac = ((ratio - 1.0).abs() > 0.02).float().mean()
            
            mini_dict_ = self.log_dict_(
                losses_value_loss=value_loss.item(),
                losses_policy_loss=pg_loss.item(),
                losses_entropy=entropy_loss.item(),
                losses_total_loss=total_loss.item(),
                losses_approx_kl=approx_kl.item(),
                losses_clipfrac=clipfrac.item(),
                train_grad_norm=grad_norm.item()
            )
        
        return mini_dict_