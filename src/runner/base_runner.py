# -*- encoding: utf-8 -*-
'''
@File       :base_runner.py
@Description:
@Date       :2025/03/27 10:15:53
@Author     :junweiluo
@Version    :python
'''
import random
import os
import time
import torch
import numpy as np
from abc import ABC

from torch.utils.tensorboard import SummaryWriter
import wandb



class BaseRunner(ABC):
    def __init__(self, config):
        """Initialize BaseRunner with experiment configuration."""
        self.all_args = config['all_args']
        self.env_id = self.all_args.env_id
        self.envs = config['envs']

        # Logger and experiment tracking
        self.setup_logging()
        self.setup_env_seeds()

        # Initialize trainer, policy, and buffer
        self.trainer = config['trainer']
        self.buffer = config['buffer']
        self.start_time = time.time()
        self.global_step = 0
        self.next_obs, self.next_done = None, None
        

    def env_reset(self):
        if self.global_step > 0:
            self.all_args.logger.warning(f"global step is {self.global_step}, there is no need to reset envs secondly!")
            return 
        # TRY NOT TO MODIFY: start the game
        next_obs, _ = self.envs.reset(seed=self.all_args.seed)
        self.next_obs = torch.Tensor(next_obs).to(self.all_args.device)
        self.next_done = torch.zeros(self.all_args.num_envs).to(self.all_args.device)
    

    def setup_logging(self):
        """Setup wandb and TensorBoard logging."""
        self.run_name = f"{self.all_args.exp_name}_seed{self.all_args.seed}_{int(time.time())}_{os.getppid()}"
        self.all_args.logger.info(f"wandb project is {self.all_args.wandb_project_name}, run name is {self.run_name}, group name is {self.all_args.exp_name  + '_pi_old'}!")
        if self.all_args.track:
            wandb.init(
                project=self.all_args.wandb_project_name,
                group=self.all_args.exp_name + "_pi_old_ent",
                sync_tensorboard=True,
                config=vars(self.all_args),
                name=self.run_name,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.all_args).items()])
            ),
        )

    def setup_env_seeds(self):
        """Set environment and torch seeds."""
        torch.manual_seed(self.all_args.seed)
        torch.backends.cudnn.deterministic = self.all_args.torch_deterministic
        np.random.seed(self.all_args.seed)
        random.seed(self.all_args.seed)


    def make_envs(self):
        """Create vectorized environments."""
        raise NotImplementedError()


    def run(self):

        
        return 0
            

    def log_episode(self, infos):
        """Log episodic return and length."""
        # for index, info in enumerate(infos["final_info"]):
        #     if info and "episode" in info:
        #         episodic_return = info["episode"]["r"]
        #         episodic_length = info["episode"]["l"]
        #         self.writer.add_scalar("charts/episodic_return", episodic_return, self.global_step)
        #         self.writer.add_scalar("charts/episodic_length", episodic_length, self.global_step)
                # self.all_args.logger.info(f"Step {self.global_step}: episodic_return = {episodic_return}, episodic_length = {episodic_length}")
        
        # mujoco-v5
        episodic_return = np.sum((infos["episode"]["r"] * infos["episode"]["_r"])) / np.sum(infos["episode"]["_r"])
        episodic_length = np.sum(infos["episode"]["l"] * infos["episode"]["_l"]) / np.sum(infos["episode"]["_l"])
        self.writer.add_scalar("charts/episodic_return", episodic_return, self.global_step)
        self.writer.add_scalar("charts/episodic_length", episodic_length, self.global_step)
        self.writer.add_scalar("charts/global_step", self.global_step)
        

        
    def collect_rollout(self):
        raise NotImplementedError
