# -*- encoding: utf-8 -*-
'''
@File    :   mujuco_main.py
@Time    :   2025/03/25 22:18:00
@Author  :   junewluo 
'''

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import torch
import gymnasium as gym
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import torch.optim as optim
from utils import get_conf
torch.set_num_threads(1)

if __name__ == "__main__":
    args = get_conf()
    config_ = {
        "all_args" : args,
        "trainer": None,
        "buffer": None,
        "envs": None,   
    }
    
    if args.env_type == "atari":
        from runner import AtariRunner as Runner
        from buffer import AtariBuffer as Buffer
        from algos.ppo_family.base.dis_policy_value import Discrete_PolicyValue as Network
        if args.algo == "appo":
            from algos.ppo_family.appo.dis_appo_trainer import Discrete_APPO_Trainer as Trainer
        elif args.algo == "ppo-clip" or args.algo == "ppo2-kl" or args.algo == "ppo2-ent":
            from algos.ppo_family.ppo_clip.dis_ppo2_trainer import Discrete_PPO2_Trainer as Trainer
        elif args.algo == "ppo-penalty":
            from algos.ppo_family.ppo_penalty.dis_ppo1_trainer import Discrete_PPOPenalty_Trainer as Trainer
        
    elif args.env_type == "mujoco":
        from runner import MujocoRunner as Runner
        from buffer import MujocoBuffer as Buffer
        from algos.ppo_family.base.con_policy_value import Continous_PolicyValue as Network
        if args.algo == "appo":
            from algos.ppo_family.appo.con_appo_trainer import Continous_APPO_Trainer as Trainer
        elif args.algo == "ppo-clip" or args.algo == "ppo2-kl" or args.algo == "ppo2-ent":
            from algos.ppo_family.ppo_clip.con_ppo2_trainer import Continous_PPO2_Trainer as Trainer
        elif args.algo == "ppo-penalty":
            from algos.ppo_family.ppo_penalty.con_ppo1_trainer import Continous_PPOPenalty_Trainer as Trainer
        elif args.algo == "spo":
            from algos.ppo_family.spo.con_spo_trainer import Continous_SPO_Trainer as Trainer
    else:
        args.logger.error(f"env_type:{args.env_type} hasn't not yet implemented! Only Support ['mujoco', 'atari]")
        sys.exit({"ExitCode": 1, "ErrorType": "NotImplementedError"})
    
    runner = Runner(config_)
    envs = gym.vector.SyncVectorEnv(
        [runner.make_envs(i) for i in range(runner.all_args.num_envs)]
    )

    if args.env_type == "mujoco" and not isinstance(envs.single_action_space, gym.spaces.Box):
        runner.all_args.logger.error(f"{args.env_type} only continuous action space is supported")
        raise TypeError(f'{type(envs.single_action_space)} != {gym.spaces.Box}')
    elif args.env_type  == "atari" and not isinstance(envs.single_action_space, gym.spaces.Discrete):
        runner.all_args.logger.error(f"{args.env_type} only discrete action space is supported")
        raise TypeError(f'{type(envs.single_action_space) != {gym.spaces.Discrete}}')
    

    # logger.info(f"env is {args.env_id}, n_rollout_thread is {args.num_envs}, sample action num is {args.sample_action_num}")
    if args.env_type == "atari":
        runner.all_args.single_observation_space = envs.single_observation_space
        runner.all_args.single_action_space = envs.single_action_space
        runner.all_args.discrete_action_space_n = envs.single_action_space.n
        runner.envs = envs
        agent = Network(num_actions = runner.all_args.discrete_action_space_n).to(args.device)
        # agent = Network(runner.envs, sample_action_num = args.sample_action_num).to(args.device)
    elif args.env_type == "mujoco":
        runner.all_args.single_observation_space = envs.single_observation_space
        runner.all_args.single_action_space = envs.single_action_space
        runner.envs = envs
        agent = Network(runner.envs, sample_action_num = args.sample_action_num).to(args.device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    trainer = Trainer(args, agent, optimizer)
    replay_buffer = Buffer(args)
    runner.trainer = trainer
    runner.buffer = replay_buffer
    runner.all_args.logger.info(f'Trainer is {runner.trainer}, Buffer is {runner.buffer}')
    
    runner.env_reset()
    runner.run()

    runner.envs.close()
    runner.writer.close()
