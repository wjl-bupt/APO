# -*- encoding: utf-8 -*-
'''
@File       :mujoco_runner.py
@Description:
@Date       :2025/03/27 10:34:53
@Author     :junweiluo
@Version    :python
'''
import torch
import time
import numpy as np
import gymnasium as gym
from .base_runner import BaseRunner
from utils import compute_advantages
from tqdm import trange

class MujocoRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
    
    def make_envs(self, idx):
        def thunk():
            if self.all_args.capture_video and idx == 0:
                env = gym.make(self.all_args.env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
            else:
                if self.all_args.env_id == "Pusher-v5":
                    custom_max_steps = 1000
                    env = gym.make(self.all_args.env_id, max_episode_steps = custom_max_steps)
                else:
                    env = gym.make(self.all_args.env_id)
            
                # env = gym.wrappers.TimeLimit(env.env, max_episode_steps = custom_max_steps)
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space = env.observation_space)
            env = gym.wrappers.NormalizeReward(env, gamma=self.all_args.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env

        return thunk

    def reshape_(self, data):
        """ reshape tensor

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        obs, logprobs, actions, advantages, returns, values, means, stds = data
        # flatten the batch
        b_obs = obs.reshape((-1,) + self.all_args.single_observation_space.shape)
        # b_logprobs shape is (args.num_steps * args.num_envs, args.sample_action_num)
        b_logprobs = logprobs.reshape((-1,) + (self.all_args.sample_action_num,))
        # b_actions shape is (args.num_steps * args.num_envs, args.sample_action_num, action_dim)
        b_actions = actions.reshape((-1,) + (self.all_args.sample_action_num,) + self.all_args.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_means = means.reshape(self.all_args.batch_size, -1)
        b_stds = stds.reshape(self.all_args.batch_size, -1)
        
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_means, b_stds

    def run(self):
        """ Main training loop."""
        for iteration in trange(1, self.all_args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.all_args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.all_args.num_iterations
                lrnow = frac * self.all_args.learning_rate
                self.trainer.optimizer.param_groups[0]["lr"] = lrnow
                
                # self.trainer.decay_delta = 1.0 - self.all_args.decay_delta * frac
                # self.trainer.decay_delta = min(0.2 + (iteration - 1) * (0.8 / (self.all_args.num_iterations // 2 - 1)), 1.0)
                # self.writer.add_scalar("charts/decay_coef", self.trainer.decay_delta)
            
            self.collect_rollout()

            obs, actions, logprobs, rewards, dones, values, means, stds = self.buffer.pop()
            returns, advantages = compute_advantages(
                args = self.all_args, 
                agent = self.trainer.agent, 
                rewards = rewards, 
                values = values, 
                next_obs = self.next_obs, 
                next_done = self.next_done, 
                dones = dones,
            )
            
            data = (obs, logprobs, actions, advantages, returns, values, means, stds)
            data = self.reshape_(data)
            
            for dict_ in self.trainer.update_one_episode(data):
                for k, v in dict_.items():
                    self.writer.add_scalar(k, v)
            
            self.writer.add_scalar("charts/var_returns", torch.var(returns).item())
            self.writer.add_scalar("losses/SPS", int(self.global_step / (time.time() - self.start_time)),)
            

    
    def collect_rollout(self):
        for step in range(0, self.all_args.num_steps):
            self.global_step += self.all_args.num_envs
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, mean_std = self.trainer.agent.get_action_and_value(self.next_obs)

            
            # TRY NOT TO MODIFY: execute the game and log data.
            obs_, reward, terminations, truncations, infos = self.envs.step(action[:,0,:].cpu().numpy())
            data = (self.next_obs, action, logprob, torch.tensor(reward).to(self.all_args.device).view(-1), self.next_done, value.flatten().detach(), mean_std.loc, mean_std.scale)
            self.buffer.push(data, step)
            self.next_obs = obs_
            self.next_done = np.logical_or(terminations, truncations)
            self.next_obs, self.next_done = torch.Tensor(self.next_obs).to(self.all_args.device), torch.Tensor(self.next_done).to(self.all_args.device)
            
            # if "final_info" in infos:
            #     self.log_episode(infos)
                
            if "episode" in infos:
                self.log_episode(infos)