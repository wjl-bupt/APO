# -*- encoding: utf-8 -*-
'''
@File       :atari_runner.py
@Description:
@Date       :2025/03/27 14:39:25
@Author     :junweiluo
@Version    :python
'''
import time
import torch
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from .base_runner import BaseRunner
from utils import compute_advantages
from tqdm import trange
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


class AtariRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
    
    # v4
    def make_envs(self, idx):
        def thunk():
            if self.all_args.capture_video and idx == 0:
                env = gym.make(self.all_args.env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
            else:
                env = gym.make(self.all_args.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if self.all_args.capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            # env = gym.wrappers.RecordEpisodeStatistics(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            # env = AtariPreprocessing(
            #     env = env,
            # )
            env = gym.wrappers.FrameStackObservation(env, 4)
            
            return env
        
        return thunk
    


    def reshape_(self, data):
        obs, actions, log_probs, advantages, returns, values, old_logits  = data
        # TODO(weijun): 添加observation_shape和discrete_action_space_n到args中
        b_obs = obs.reshape((-1, ) + self.all_args.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_log_probs = log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_old_logits = old_logits.reshape((-1, self.all_args.discrete_action_space_n))
        
        return b_obs, b_actions, b_log_probs, b_returns, b_advantages, b_values, b_old_logits


    def run(self):
        for iteration in trange(1, self.all_args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.all_args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.all_args.num_iterations
                lrnow = frac * self.all_args.learning_rate
                self.trainer.optimizer.param_groups[0]["lr"] = lrnow
                self.trainer.decay_delta = frac * self.all_args.decay_delta

            self.collect_rollout()
                            
            obs, actions, rewards, dones, logprobs, values, total_logits = self.buffer.pop()

            returns, advantages = compute_advantages(
                args = self.all_args, 
                agent = self.trainer.agent, 
                rewards = rewards, 
                values = values, 
                next_obs = self.next_obs, 
                next_done = self.next_done, 
                dones = dones
            )
              
            # b_obs, b_actions, b_log_probs, b_returns, b_advantages, b_values, b_old_logits
            data = (obs, actions, logprobs, returns, advantages, values, total_logits)
            data = self.reshape_(data)
            
            for dict_ in self.trainer.update_one_episode(data):
                for k,v in dict_.items():
                    self.writer.add_scalar(k, v)
            
            self.writer.add_scalar("losses/SPS", int(self.global_step / (time.time() - self.start_time)))
    
    def collect_rollout(self):
        for step in range(0, self.all_args.num_steps):
            self.global_step += self.all_args.num_envs
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, total_logits = self.trainer.agent.get_action_and_value(self.next_obs)
            
            # TRY NOT TO MODIFY: execute the game and log data.
            obs_, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            self.buffer.push(self.next_obs, action, torch.tensor(reward).to(self.all_args.device).view(-1), self.next_done, logprob, value.flatten(), total_logits, step)
            self.next_obs = obs_
            self.next_done = np.logical_or(terminations, truncations)
            self.next_obs, self.next_done = torch.Tensor(self.next_obs).to(self.all_args.device), torch.Tensor(self.next_done).to(self.all_args.device)

            if "episode" in infos:
                self.log_episode(infos = infos)
