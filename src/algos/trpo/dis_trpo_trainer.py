# -*- encoding: utf-8 -*-
'''
@File       :dis_trpo_trainer.py
@Description:Discrete TRPO (Trust Region Policy Optimization) Algorithm Trainer
@Date       :2025/04/24
@Author     :Trae
@Version    :python
'''

import torch
import torch.nn as nn
import numpy as np
from .trpo_trainer import TRPOTrainer


class DiscreteTRPOTrainer(TRPOTrainer):
    def __init__(self, args, agent, optimizer):
        super().__init__(args, agent, optimizer)
    
    # Inherits all methods from TRPOTrainer
    # Can override specific methods if needed for discrete environments