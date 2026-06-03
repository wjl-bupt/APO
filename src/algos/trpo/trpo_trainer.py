
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
import torch.optim as optim
from ..ppo_family.base.base_trainer import BaseTrainer

def compute_kl_divergence(self, obs, actions, old_logprobs):
    """
    Compute KL divergence between old and new policies
    """
    with torch.no_grad():
        _, new_logprob, _, _, _ = self.agent.get_action_and_value(obs, actions)
        kl_div = old_logprobs - new_logprob
    return kl_div

# def fisher_vector_product(self, vector, obs, actions, logprobs):
#     """
#     Compute Fisher Information Matrix-vector product
#     """
#     kl = self.compute_kl_divergence(obs, actions, logprobs)
#     kl = kl.mean()
    
#     # First gradient
#     grads = torch.autograd.grad(kl, self.agent.parameters(), create_graph=False)
#     flat_grad_kl = torch.cat([grad.flatten() for grad in grads])
    
#     # Second gradient (Fisher-vector product)
#     grad_vector_product = torch.dot(flat_grad_kl, vector)
#     fisher_vector = torch.autograd.grad(grad_vector_product, self.agent.parameters())
#     flat_fisher_vector = torch.cat([grad.flatten() for grad in fisher_vector])
    
#     return flat_fisher_vector + self.cg_damping * vector


class TRPOTrainer(BaseTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
        self.max_kl = args.max_kl if hasattr(args, 'max_kl') else 0.01
        self.cg_damping = args.cg_damping if hasattr(args, 'cg_damping') else 0.1
        self.line_search_coef = args.line_search_coef if hasattr(args, 'line_search_coef') else 0.8
        self.max_backtracks = args.max_backtracks if hasattr(args, 'max_backtracks') else 10
        self.value_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=2.5e-4)
        self.optimizer = torch.optim.Adam(list(self.agent.actor_mean.parameters()) + [self.agent.actor_logstd], lr=2.5e-4)

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

    # def fisher_vector_product(self, vector, obs, actions, logprobs):
    #     """
    #     Compute Fisher Information Matrix-vector product
    #     """
    #     kl = self.agent.compute_kl_divergence(obs, actions, logprobs)
    #     kl = kl.mean()
        
    #     # First gradient
    #     grads = torch.autograd.grad(kl, self.agent.parameters(), create_graph=False)
    #     flat_grad_kl = torch.cat([grad.flatten() for grad in grads])
        
    #     # Second gradient (Fisher-vector product)
    #     grad_vector_product = torch.dot(flat_grad_kl, vector)
    #     fisher_vector = torch.autograd.grad(grad_vector_product, self.agent.parameters())
    #     flat_fisher_vector = torch.cat([grad.flatten() for grad in fisher_vector])
        
    #     return flat_fisher_vector + self.cg_damping * vector

    def update_one_episode(self, data):
        """
        TRPO (actor) + minibatch critic update
        """

        b_obs, b_logprobs, b_actions, advantages, b_returns, b_values, means, stds = data
        ratio_stats = []
        kl_stats = []
        policy_losses = []
        value_losses = []
        entropy_stats = []

        min_ratio, max_ratio = 10.0, 0.0
        # =========================================================
        # 0. ADVANTAGE
        # =========================================================
        advantages = b_returns - b_values

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = advantages.detach()

        # =========================================================
        # 1. OLD POLICY (for KL constraint)
        # =========================================================
        old_means = means.detach()
        old_stds = stds.detach()

        batch_size = b_obs.shape[0]
        b_inds = torch.randperm(batch_size, device=b_obs.device)

        policy_params = list(self.agent.actor_mean.parameters()) + [self.agent.actor_logstd]

        # =========================================================
        # 2. LOG PROB FUNCTION
        # =========================================================
        def log_prob_gaussian(a, mu, std):
            var = std ** 2
            return -0.5 * (((a - mu) ** 2) / var + torch.log(var) + 2 * np.log(2 * np.pi)).sum(-1)

        # =========================================================
        # 3. POLICY GRADIENT (MINI-BATCH)
        # =========================================================
        policy_grads = []

        for start in range(0, batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mb_inds = b_inds[start:end]

            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]
            mb_adv = advantages[mb_inds]

            new_means, _ = self.agent.get_dist_mean_and_std(mb_obs)
            new_logstd = self.agent.actor_logstd.expand_as(new_means)
            new_stds = torch.exp(new_logstd)

            new_logprob = log_prob_gaussian(mb_actions, new_means, new_stds)
            ratio = torch.exp(new_logprob - mb_logprobs)

            pg_loss = (ratio * mb_adv).mean()

            self.optimizer.zero_grad()
            pg_loss.backward()

            policy_grads.append(self.flatten_grad(policy_params).detach())

        policy_gradient = torch.stack(policy_grads).mean(0)

        # =========================================================
        # 4. KL FUNCTION (FULL BATCH)
        # =========================================================
        def kl_fn():
            new_means, new_logstd = self.agent.get_dist_mean_and_std(b_obs)
            new_stds = torch.exp(new_logstd)

            kl = (
                torch.log(new_stds / old_stds)
                + (old_stds ** 2 + (old_means - new_means) ** 2) / (2.0 * new_stds ** 2)
                - 0.5
            ).sum(-1).mean()

            return kl

        kl = kl_fn().detach()

        # =========================================================
        # 5. FISHER VECTOR PRODUCT
        # =========================================================
        def fisher_vector_product(v):

            new_means, new_logstd = self.agent.get_dist_mean_and_std(b_obs)
            new_stds = torch.exp(new_logstd)

            kl = (
                torch.log(new_stds / old_stds)
                + (old_stds ** 2 + (old_means - new_means) ** 2) / (2.0 * new_stds ** 2)
                - 0.5
            ).sum(-1).mean()

            grads = torch.autograd.grad(
                kl,
                policy_params,
                create_graph=True,
            )

            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            gv = torch.dot(flat_grad, v)
            hvp = torch.autograd.grad(gv, policy_params)

            return torch.cat([g.reshape(-1) for g in hvp]) + self.cg_damping * v

        # =========================================================
        # 6. CONJUGATE GRADIENT
        # =========================================================
        stepdir = self.conjugate_gradient(
            fisher_vector_product,
            policy_gradient,
            10
        )

        shs = 0.5 * torch.dot(stepdir, fisher_vector_product(stepdir))
        step_size = torch.sqrt(2 * self.max_kl / (shs + 1e-8))
        full_step = stepdir * step_size

        # =========================================================
        # 7. LINE SEARCH
        # =========================================================
        old_params = self.flatten_parameters(policy_params).clone()

        def surrogate_loss(obs, actions, logprobs, adv):
            new_means, _ = self.agent.get_dist_mean_and_std(obs)
            new_logstd = self.agent.actor_logstd.expand_as(new_means)
            new_stds = torch.exp(new_logstd)

            new_logprob = log_prob_gaussian(actions, new_means, new_stds)
            ratio = torch.exp(new_logprob - logprobs)

            return -(ratio * adv).mean()

        success = False
        best_loss = float("inf")

        for j in range(int(self.max_backtracks)):

            coeff = self.line_search_coef ** j
            new_params = old_params + coeff * full_step

            self._set_flat_params(policy_params, new_params)

            with torch.no_grad():

                test_loss = surrogate_loss(
                    b_obs, b_actions, b_logprobs, advantages
                )

                test_means, test_logstd = self.agent.get_dist_mean_and_std(b_obs)
                test_stds = torch.exp(test_logstd)

                var0 = old_stds ** 2
                var1 = test_stds ** 2

                kl_test = (
                    torch.log(test_stds / old_stds)
                    + (var0 + (old_means - test_means) ** 2) / (2.0 * var1)
                    - 0.5
                ).sum(-1).mean()

            if kl_test <= self.max_kl and test_loss <= best_loss:
                success = True
                best_loss = test_loss.item()
                break

        if not success:
            self._set_flat_params(policy_params, old_params)

        # =========================================================
        # 8. VALUE FUNCTION UPDATE (MINI-BATCH SGD)
        # =========================================================
        value_losses = []

        for start in range(0, batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mb_inds = b_inds[start:end]

            new_value = self.agent.get_value(b_obs[mb_inds])

            value_loss = ((new_value - b_returns[mb_inds]) ** 2).mean()
            value_losses.append(value_loss.detach().item())

        # value_loss = torch.stack(value_losses).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # =========================================================
        # 9. ENTROPY
        # =========================================================
        with torch.no_grad():
            new_means, new_logstd = self.agent.get_dist_mean_and_std(b_obs)
            entropy = new_logstd.sum(-1).mean()

        mini_dict_ = self.log_dict_(
            losses_policy_loss=pg_loss.mean().item(),
            losses_value_loss=value_loss.mean().item(),
            losses_entropy=entropy.mean().item(),
            # TRPO core
            losses_kl=kl.item(),
        )
        yield mini_dict_
        # # =========================================================
        # # 10. LOGGING
        # # =========================================================
        # return self.log_dict_(
        #     losses_value_loss=value_loss.item(),
        #     losses_policy_loss=best_loss,
        #     losses_entropy=entropy.item(),
        #     losses_approx_kl=kl.item(),
        #     train_step_successful=success
        # )

    def _set_flat_params(self, parameters, flat_params):
        """
        Set parameters to flattened values
        """
        idx = 0
        for param in parameters:
            num_params = param.numel()
            param.data.copy_(flat_params[idx:idx+num_params].view(param.shape))
            idx += num_params