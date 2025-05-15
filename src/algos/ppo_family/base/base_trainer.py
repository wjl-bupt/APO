# -*- encoding: utf-8 -*-
'''
@File       :base_trainer.py
@Description:
@Date       :2025/03/31 10:40:11
@Author     :junweiluo
@Version    :python
'''
import os
import torch
import sys
from prettytable import PrettyTable

class BaseTrainer(object):
    def __init__(self, args, agent, optimizer):
        
        # common parameters
        self.env_type = args.env_type
        self.logger = args.logger
        self.max_grad_norm = args.max_grad_norm
        self.update_epochs = args.update_epochs
        self.norm_adv = args.norm_adv
        self.clip_vloss = args.clip_vloss
        self.vf_coef = args.vf_coef
        self.target_kl = args.target_kl
        self.batch_size = args.batch_size
        self.mini_batch_size = args.minibatch_size
        self.sample_action_num = args.sample_action_num
        # self.single_action_space_n = args.single_action_space.shape[0]
        self.ent_coef = args.ent_coef
        self.clip_coef = args.clip_coef
        
        self.agent = agent
        self.optimizer = optimizer
        self.batch_index = 0
        
    

    def dis_params(self):
        class_name = self.__class__.__name__ 
        table = PrettyTable([f"{class_name}.attr", "value", "type"])
        for key, value in self.__dict__.items():
            table.add_row([key, value, type(value)])
        self.logger.success(f'class {class_name} init successfully!\n{table}') 

    def log_dict_(self, **kwargs):
        mini_dict_ = {}
        for key, value in kwargs.items():
            if "imp_weight" in key:
                log_key = key.replace("_","$",1).replace("_","/",1).replace("$","_")
            else:
                log_key = key.replace("_","/",1)
            mini_dict_[log_key] = value
        return mini_dict_

    def compute_value_loss(self,  mb_returns, mb_values, newvalue):
        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        
        return v_loss  


    def compute_dis_ratios_family(self, **kwargs):
        new_log_prob, mb_log_probs, new_logits, mb_old_logits, mb_actions \
            = kwargs["new_log_prob"], kwargs["mb_log_probs"], kwargs["new_logits"], kwargs["mb_old_logits"], kwargs["mb_actions"]
        
        # ppo-clip ratio
        log_probs = new_log_prob - mb_log_probs
        ratio1 = log_probs.exp()
        # caculate other actions ratio
        mask_ = torch.ones_like(mb_old_logits, dtype=torch.bool)
        mask_[torch.arange(mb_old_logits.shape[0]), mb_actions] = 0.0
        old_logprobs = mb_old_logits[mask_].reshape(mb_old_logits.size(0), -1)
        new_logprobs = new_logits[mask_].reshape(mb_old_logits.size(0), -1)
        log_ratio2 = new_logprobs - old_logprobs
        ratio2 = log_ratio2.exp()

        return ratio1, ratio2, old_logprobs
    
    def compute_con_ratios_family(self, **kwargs):
        newlogprob, mb_logprobs = kwargs["newlogprob"], kwargs["mb_logprobs"]
        total_logratio = newlogprob - mb_logprobs
        logratio1 = total_logratio[:,0]
        ratio1 = logratio1.exp()
        logratio2 = total_logratio[:,1:]
        ratio2 = logratio2.exp()
        
        return ratio1, ratio2

    def compute_ratios_family(self, **kwargs):
        
        map_ = {
            "atari": self.compute_dis_ratios_family,
            "mujoco": self.compute_con_ratios_family,
        }
        
        func = map_.get(self.env_type, self._not_implemented)
        return func(**kwargs)
    
    def compute_policy_loss(self):
        self.logger.error("function: compute_policy_loss had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def ppo_update(self):
        self.logger.error("function: ppo_update had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def update_one_episode(self):
        self.logger.error("function: update_one_episode had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def _not_implemented(self, *args):
        raise NotImplementedError()
    
    
    
    
        