# -*- encoding: utf-8 -*-
'''
@File       :dis_ppo_kl_trainer.py
@Description:
@Date       :2025/04/01 09:38:38
@Author     :junweiluo
@Version    :python
'''

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from ..base.base_trainer import BaseTrainer

class Discrete_PPOPenalty_Trainer(BaseTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
        self.penalty_coef = args.penalty_coef
        self.num_alter_logprobs = min(args.sample_action_num, args.discrete_action_space_n.item())
    
    def compute_policy_loss(self, mb_advantages, ratio1, kl):
        pg_loss_1 = (-mb_advantages * ratio1).mean()
        # pg_loss2 = -mb_advantages * torch.clamp(ratio1, (1 - self.clip_coef), (1 + self.args.clip_coef))
        # pg_loss_1 = torch.max(pg_loss1, pg_loss2).mean()
        pg_loss = pg_loss_1 + self.penalty_coef * kl
        
        return pg_loss, pg_loss_1.item(), self.penalty_coef * kl
        
    
    def update_one_episode(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Yields:
            _type_: _description_
        """
        
        b_inds = np.arange(self.batch_size)
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds) 
            for dict_ in self.ppo_update(data = data, b_inds = b_inds, epoch = epoch):
                if "kl" in dict_:
                    # dynamic adjust penalty coefficience
                    kl = dict_["kl"]
                    if kl > 1.5 * self.target_kl:
                        self.penalty_coef *= 2
                    elif kl < self.target_kl / 1.5:
                        self.penalty_coef /= 2
                    yield {"charts/kl_coef": self.penalty_coef}
                else:
                    yield dict_ 
        
        
    def ppo_update(self, data, b_inds, epoch):
        b_obs, b_actions, b_log_probs, b_returns, b_advantages, b_values, b_old_logits = data

        np.random.shuffle(b_inds)
        ratio1_clipfracs, ratio1_devations = 0.0, 0.0
        ratio2_devations = 0.0
        min_ratio1, max_ratio1 =  10.0, 0.0
        min_ratio2, max_ratio2 = 10.0, 0.0
        total_size = 0

        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mb_inds = b_inds[start:end]

            # The latest outputs of the policy network and value network
            _, new_log_prob, new_entropy, new_value, new_logits = self.agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )

            # Probability ratio
            ratio1, ratio2, _ = self.compute_ratios_family(
                new_log_prob = new_log_prob, 
                mb_log_probs = b_log_probs[mb_inds], 
                new_logits = new_logits, 
                mb_old_logits = b_old_logits[mb_inds], 
                mb_actions = b_actions[mb_inds]
            )

            # Advantage normalization
            mb_advantages = b_advantages[mb_inds]
            # if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # 计算KL
            kl = torch.distributions.kl.kl_divergence(
                Categorical(logits=b_old_logits[mb_inds]), 
                Categorical(logits=new_logits),
            ).mean()

            # Policy loss
            pg_loss, pg_loss_1, pg_loss_2 = self.compute_policy_loss(mb_advantages = mb_advantages, ratio1 = ratio1, kl = kl)
            # Value loss
            v_loss = self.compute_value_loss(mb_returns = b_returns[mb_inds], newvalue = new_value, mb_values = b_values[mb_inds])
            # Policy entropy
            entropy_loss = new_entropy.mean()
            # Total loss
            loss = pg_loss + v_loss * self.vf_coef - entropy_loss * self.ent_coef
            
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                logratio1 = ratio1.detach().log()
                old_approx_kl = (-logratio1).mean()
                approx_kl = ((ratio1 - 1) - logratio1).mean()
  
                # ratios family
                min_ratio1, max_ratio1 = min(np.min(ratio1.detach().cpu().numpy()), min_ratio1), max(np.max(ratio1.detach().cpu().numpy()), max_ratio1)
                min_ratio2, max_ratio2 = min(np.min(ratio2.detach().cpu().numpy()), min_ratio2), max(np.max(ratio2.detach().cpu().numpy()), max_ratio2)
                ratio1_clipfracs += (torch.abs(ratio1.detach().cpu() - 1.0) < self.clip_coef).float().sum()
                ratio1_devations += (torch.abs(ratio1.detach().cpu() - 1.0)).float().sum() 
                ratio2_devations += (torch.abs(ratio2.detach().cpu() - 1.0)).float().sum()  
                total_size += ratio1.shape[0]

                mini_dict_ = self.log_dict_(
                    losses_old_approx_kl = old_approx_kl.item(),
                    losses_approx_kl = approx_kl.item(),
                    losses_kl_div = kl,
                    imp_weight_ratio1 = np.mean(ratio1.detach().cpu().numpy()),
                    imp_weight_ratio2 = np.mean(ratio2.detach().cpu().numpy()),

                )
                                  
                yield mini_dict_
            
            if self.target_kl != None and kl > 4 * self.target_kl:
                # this in google's paper
                grad = None
                break
        
            # param update
            self.optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.batch_index += 1

        dict_ = self.log_dict_(
            losses_grad_norm = grad,    
            imp_weight_min_ratio1 = min_ratio1,
            imp_weight_max_ratio1 = max_ratio1,
            losses_ratio2_devations = ratio2_devations / total_size,
            losses_ratio1_clifracs =  ratio1_clipfracs / total_size,
            losses_ratio1_devations = ratio1_devations / total_size,
        )

        # last epoch
        if epoch == self.update_epochs - 1:
            yield {"kl": kl.mean()}
            
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if grad != None:
                final_dict_ = self.log_dict_(
                        losses_pg_loss_1 = pg_loss_1,
                        losses_pg_loss_2 = pg_loss_2,
                        losses_pg_loss = pg_loss.item(),
                        losses_grad_norm = grad,
                        losses_entropy = entropy_loss.item(),
                        losses_explained_variance = explained_var,
                    )
            else:
                final_dict_ = self.log_dict_(
                        losses_pg_loss_1 = pg_loss_1,
                        losses_pg_loss_2 = pg_loss_2,
                        losses_pg_loss = pg_loss.item(),
                        losses_entropy = entropy_loss.item(),
                        losses_explained_variance = explained_var,
                    )
            
            dict_ = {**dict_, **final_dict_}    
        
        yield dict_