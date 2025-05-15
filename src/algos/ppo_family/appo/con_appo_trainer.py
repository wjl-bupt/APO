# -*- encoding: utf-8 -*-
'''
@File       :con_appo_trainer.py
@Description:
@Date       :2025/03/31 15:00:18
@Author     :junweiluo
@Version    :python
'''

import torch
import torch.nn as nn
import numpy as np
from ..base.base_trainer import BaseTrainer
from utils import compute_kld

class Continous_APPO_Trainer(BaseTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
        self.clip_coef = args.clip_coef
        self.decay_delta = args.decay_delta
        
    
    def ppo_update(self, data, b_inds, epoch):
        """ ppo update epochs

        Args:
            data (_type_): _description_
            b_inds (_type_): _description_
            epoch (_type_): _description_

        Yields:
            _type_: _description_
        """
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_means, b_stds = data
        ratio1_clipfracs, ratio2_clipfracs = 0.0, 0.0
        min_ratio1, max_ratio1 =  10.0, 0.0
        min_ratio2, max_ratio2 = 10.0, 0.0
        
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mb_inds = b_inds[start:end]
            _, newlogprob, entropy, newvalue, new_mean_std = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            ratio1, ratio2 = self.compute_ratios_family(
                newlogprob = newlogprob, 
                mb_logprobs = b_logprobs[mb_inds]
            )
            
            mb_advantages = b_advantages[mb_inds]
            if self.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            # self.logger.info(f"adv mean:{torch.mean(mb_advantages)}, adv min:{torch.min(mb_advantages)}, adv max:{torch.max(mb_advantages)}")
            # Policy loss
            pg_loss, pg_loss_1, pg_loss_2 = self.compute_policy_loss(
                mb_advantages = mb_advantages, 
                ratio1 = ratio1, 
                ratio2 = ratio2,
                mb_old_logprobs = b_logprobs[mb_inds],
                mb_returns_vars = torch.var(b_returns[mb_inds]).item(),
                new_logprobs = newlogprob,
            )
            # value loss
            v_loss = self.compute_value_loss(mb_returns = b_returns[mb_inds], mb_values = b_values[mb_inds], newvalue = newvalue)
            # entropy loss
            entropy_loss = entropy.mean()
            # total loss
            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
            # param update
            self.optimizer.zero_grad()
            loss.backward()
            # grad clip
            grad = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                logratio1 = ratio1.log()
                old_approx_kl = (-logratio1).mean()
                approx_kl = ((ratio1 - 1) - logratio1).mean()
                # use mean and std to calculate kl
                new_mean = new_mean_std.loc
                new_std = new_mean_std.scale
                kl = compute_kld(b_means[mb_inds], b_stds[mb_inds], new_mean, new_std)
                # ratios family
                min_ratio1, max_ratio1 = min(np.min(ratio1.detach().cpu().numpy()), min_ratio1), max(np.max(ratio1.detach().cpu().numpy()), max_ratio1)
                min_ratio2, max_ratio2 = min(np.min(ratio2.detach().cpu().numpy()), min_ratio2), max(np.max(ratio2.detach().cpu().numpy()), max_ratio2)
                ratio1_clipfracs += (torch.abs(ratio1.detach().cpu() - 1.0) < self.clip_coef).float().sum()
                ratio2_clipfracs += (torch.abs(ratio2.detach().cpu() - 1.0) < self.clip_coef).float().sum()
                
                # log data for every mini-batch data                    
                mini_dict_ =  self.log_dict_(
                    losses_old_approx_kl=old_approx_kl.item(),
                    losses_approx_kl=approx_kl.item(),
                    losses_kl=kl.mean().detach().cpu().item(),
                    imp_weight_ratio1=np.mean(ratio1.detach().cpu().numpy()),
                    imp_weight_ratio2=np.mean(ratio2.detach().cpu().numpy()),
                )
                yield mini_dict_            

            self.batch_index += 1
            
        # log data for every update_epochs
        dict_ = self.log_dict_(
            imp_weight_min_ratio1 = min_ratio1,
            imp_weight_max_ratio1 = max_ratio1,
            imp_weight_min_ratio2 = min_ratio2,
            imp_weight_max_ratio2 = max_ratio2,
            losses_ratio1_clifracs = ratio1_clipfracs / self.batch_size,
            losses_ratio2_clifracs = ratio2_clipfracs / self.batch_size,
        )
        # last epoch
        if epoch == self.update_epochs - 1:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            final_dict_ = self.log_dict_(
                    losses_pg_loss_1 = pg_loss_1,
                    losses_pg_loss_2 = pg_loss_2,
                    losses_pg_loss = pg_loss.item(),
                    losses_grad_norm = grad,
                    losses_entropy = entropy_loss.item(),
                    losses_explained_variance = explained_var,
                )
            dict_ = {**dict_, **final_dict_}
            
        yield dict_            
    
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
                yield dict_
        
    def compute_policy_loss(self, mb_advantages, ratio1, ratio2, **kwargs):
        """ compute policy loss

        Args:
            mb_advantages (_torch.Tensor_): GAE adavantages.
            ratio1 (_torch.Tensor_): Probability ratio of old and new policy for environmental sampling actions

        Returns:
            _type_: _description_
        """

        

        # pg_loss1 = -mb_advantages * ratio1
        # pg_loss2 = -mb_advantages * torch.clamp(ratio1, (1 - self.clip_coef), (1 + self.clip_coef))
        # pg_loss_1 = torch.max(pg_loss1, pg_loss2).mean()

        
        # # v1: 0.5 * \pi_{old} * (r1 - 1)**2
        # mb_old_logprobs = kwargs["mb_old_logprobs"]
        # pg_loss_2 = (0.5 * mb_old_logprobs[:,1:].exp() * (ratio2 - 1)**2).mean()

        # v2: 0.5 * |A|* \pi_{old} * (r1 - 1)**2
        # mb_old_probs_r2 = kwargs["mb_old_logprobs"][:,1:].exp()
        # mb_advantages_ = mb_advantages.unsqueeze(1).expand(-1, mb_old_probs_r2.shape[1])
        # pg_loss_2 = (0.5 * torch.abs(mb_advantages_) * mb_old_probs_r2 * (ratio2 - 1)**2).mean() / self.single_action_space_n
        
        
        # v3: 0.5 * |A|* \pi_{old} * (r1 - 1)**2 / d
        # mb_old_probs_r2 = kwargs["mb_old_logprobs"][:,1:].exp()
        # mb_advantages_ = mb_advantages.unsqueeze(1).expand(-1, mb_old_probs_r2.shape[1])
        # pg_loss_2 = (0.5 * torch.abs(mb_advantages_) * mb_old_probs_r2 * (ratio2 - 1)**2).mean() / self.single_action_space_n
    
        # v4: 0.5 * \pi_{old} * (r1 - 1)**2
        # mb_old_logprobs = kwargs["mb_old_logprobs"]
        # pg_loss_2 = (mb_old_logprobs[:,1:].exp() * (ratio2 - 1)**2).mean()
    
        # v5: 0.5 * \pi_{old} * (r1 - 1)**2
        # mb_old_logprobs = kwargs["mb_old_logprobs"]
        # pg_loss_2 = self.decay_delta * (0.5 * mb_old_logprobs[:,1:].exp() * (ratio2 - 1)**2).mean()
        
        # v6: 0.5 * var_returns * \pi_{old} * (r1 - 1)**2
        # mb_old_logprobs = kwargs["mb_old_logprobs"]
        # mb_returns_vars = kwargs["mb_returns_vars"]
        # pg_loss_2 = (mb_returns_vars * 0.5 * mb_old_logprobs[:,1:].exp() * (ratio2 - 1)**2).mean()
        
        
        # pg_loss1 = -mb_advantages * ratio1
        # pg_loss2 = -mb_advantages * torch.clamp(ratio1, (1 - self.clip_coef), (1 + self.clip_coef))
        # pg_loss_1 = torch.max(pg_loss1, pg_loss2).mean()
        # mb_old_logprobs = kwargs["mb_old_logprobs"]
        # pg_loss_2 = (0.5 * mb_old_logprobs[:,1:].exp() * (ratio2 - 1)**2).mean()
        
        pg_loss1 = -mb_advantages * ratio1
        pg_loss2 = -mb_advantages * torch.clamp(ratio1, (1 - self.clip_coef), (1 + self.clip_coef))
        pg_loss_1 = torch.max(pg_loss1, pg_loss2).mean()
        mb_old_logprobs = kwargs["mb_old_logprobs"]
        # new_logprobs = kwargs["new_logprobs"]
        # ratios = (new_logprobs - mb_old_logprobs).exp()
        pg_loss_2 = (0.5 * mb_old_logprobs.exp()[:,0] *(ratio1 - 1)**2).mean()

        pg_loss = pg_loss_1  +  pg_loss_2
        
        return pg_loss, pg_loss_1.item(), pg_loss_2.item()
        
    
    