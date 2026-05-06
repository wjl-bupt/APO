
# -*- encoding: utf-8 -*-
'''
@File       :trpo_trainer.py
@Description:TRPO (Trust Region Policy Optimization) Algorithm Trainer
@Date       :2025/04/24
@Author     :Trae
@Version    :python
'''

import torch
import torch.nn as nn
import numpy as np
from ..ppo_family.base.base_trainer import BaseTrainer

def compute_kl_divergence(self, obs, actions, old_logprobs):
    """
    Compute KL divergence between old and new policies
    """
    with torch.no_grad():
        _, new_logprob, _, _, _ = self.agent.get_action_and_value(obs, actions)
        kl_div = old_logprobs - new_logprob
    return kl_div

def fisher_vector_product(self, vector, obs, actions, logprobs):
    """
    Compute Fisher Information Matrix-vector product
    """
    kl = self.compute_kl_divergence(obs, actions, logprobs)
    kl = kl.mean()
    
    # First gradient
    grads = torch.autograd.grad(kl, self.agent.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.flatten() for grad in grads])
    
    # Second gradient (Fisher-vector product)
    grad_vector_product = torch.dot(flat_grad_kl, vector)
    fisher_vector = torch.autograd.grad(grad_vector_product, self.agent.parameters())
    flat_fisher_vector = torch.cat([grad.flatten() for grad in fisher_vector])
    
    return flat_fisher_vector + self.cg_damping * vector


class TRPOTrainer(BaseTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
        self.max_kl = args.max_kl if hasattr(args, 'max_kl') else 0.01
        self.cg_damping = args.cg_damping if hasattr(args, 'cg_damping') else 0.1
        self.line_search_coef = args.line_search_coef if hasattr(args, 'line_search_coef') else 0.8
        self.max_backtracks = args.max_backtracks if hasattr(args, 'max_backtracks') else 10

    def conjugate_gradient(self, A, b, nsteps, residual_tol=1e-10):
        """
        Conjugate Gradient method to solve Ax = b
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            Ap = A(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def flatten_grad(self, parameters):
        """
        Flatten gradients of parameters
        """
        return torch.cat([param.grad.flatten() for param in parameters])

    def flatten_parameters(self, parameters):
        """
        Flatten parameters
        """
        return torch.cat([param.flatten() for param in parameters])

    def unflatten_grad(self, flat_grad, parameters):
        """
        Unflatten gradients back to original shape
        """
        idx = 0
        for param in parameters:
            num_params = param.numel()
            param.grad = flat_grad[idx:idx+num_params].view(param.shape)
            idx += num_params

    def fisher_vector_product(self, vector, obs, actions, logprobs):
        """
        Compute Fisher Information Matrix-vector product
        """
        kl = self.agent.compute_kl_divergence(obs, actions, logprobs)
        kl = kl.mean()
        
        # First gradient
        grads = torch.autograd.grad(kl, self.agent.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.flatten() for grad in grads])
        
        # Second gradient (Fisher-vector product)
        grad_vector_product = torch.dot(flat_grad_kl, vector)
        fisher_vector = torch.autograd.grad(grad_vector_product, self.agent.parameters())
        flat_fisher_vector = torch.cat([grad.flatten() for grad in fisher_vector])
        
        return flat_fisher_vector + self.cg_damping * vector

    def update(self, data):
        """
        Perform one update step for TRPO
        """
        b_obs, b_actions, b_returns, b_values, b_logprobs = data
        
        # Calculate advantages
        advantages = b_returns - b_values
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy gradient
        old_params = self.flatten_parameters(self.agent.parameters()).clone()
        
        # Forward pass to get log probabilities and values
        _, new_logprob, entropy, new_value, _ = self.agent.get_action_and_value(b_obs, b_actions)
        
        # Calculate policy loss (negative of expected advantage)
        ratio = torch.exp(new_logprob - b_logprobs)
        pg_loss = -(advantages.detach() * ratio).mean()
        
        # Compute gradient
        self.optimizer.zero_grad()
        pg_loss.backward()
        policy_gradient = self.flatten_grad(self.agent.parameters()).detach()
        
        # Compute Fisher Information Matrix-vector product
        fvp = lambda v: self.fisher_vector_product(v, b_obs, b_actions, b_logprobs)
        
        # Solve for step direction using conjugate gradient
        stepdir = self.conjugate_gradient(fvp, policy_gradient, 10)
        
        # Compute step size
        shs = 0.5 * torch.dot(stepdir, self.fisher_vector_product(stepdir, b_obs, b_actions, b_logprobs))
        lagrange_multiplier = torch.sqrt(2 * self.max_kl / shs)
        step = stepdir / lagrange_multiplier
        
        # Line search to find best step size
        old_loss = pg_loss.item()
        current_params = self.flatten_parameters(self.agent.parameters()).clone()
        
        success = False
        for j in range(self.max_backtracks):
            # Update parameters
            new_params = current_params - self.line_search_coef ** j * step
            self._set_flat_params(self.agent.parameters(), new_params)
            
            # Evaluate new policy
            with torch.no_grad():
                _, new_logprob_new, _, new_value_new, _ = self.agent.get_action_and_value(b_obs, b_actions)
                ratio_new = torch.exp(new_logprob_new - b_logprobs)
                new_pg_loss = -(advantages.detach() * ratio_new).mean()
                
                # Check if improvement is sufficient
                if new_pg_loss <= old_loss:
                    success = True
                    break
        
        if not success:
            # If line search failed, revert to original parameters
            self._set_flat_params(self.agent.parameters(), old_params)
        
        # Compute final value loss
        value_loss = self.compute_value_loss(b_returns, b_values, new_value)
        
        # Log metrics
        with torch.no_grad():
            _, final_logprob, final_entropy, final_value, _ = self.agent.get_action_and_value(b_obs, b_actions)
            final_ratio = torch.exp(final_logprob - b_logprobs)
            approx_kl = ((final_ratio - 1) - (final_logprob - b_logprobs)).mean()
            clipfrac = ((final_ratio - 1.0).abs() > 0.02).float().mean()
            
            mini_dict_ = self.log_dict_(
                losses_value_loss=value_loss.item(),
                losses_policy_loss=new_pg_loss.item() if success else old_loss,
                losses_entropy=final_entropy.mean().item(),
                losses_approx_kl=approx_kl.item(),
                losses_clipfrac=clipfrac.item(),
                train_step_successful=success
            )
        
        return mini_dict_

    def _set_flat_params(self, parameters, flat_params):
        """
        Set parameters to flattened values
        """
        idx = 0
        for param in parameters:
            num_params = param.numel()
            param.data.copy_(flat_params[idx:idx+num_params].view(param.shape))
            idx += num_params