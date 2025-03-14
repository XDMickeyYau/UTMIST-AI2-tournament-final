from functools import partial
import os
import shutil

import numpy as np
from stable_baselines3 import PPO
from DodgeBasedAgent import DodgeBasedAgent
from agents import BasedBetterAgent, BasedBetterAgent, ConstantAgent, RandomAgent, RecurrentPPOAgent, SB3Agent, BasedAgent
from brawl_env import WarehouseBrawl
from camera import CameraResolution
from evaluation import evaluate_against_pool
from reward import reward_functions, signal_subscriptions
from reward_manager import RewardManager
from run_agent import TrainLogging, run_match, train
from self_play import SelfPlayDynamic, SelfPlayLatest, SelfPlayRandom, SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode, SelfPlayHandler
from imitation.data.types import Transitions
from imitation.data import rollout


def collect_transitions(num_steps=10000) -> Transitions:
    # Initialize storage
    obs_list, acts_list, dones, infos, next_obs = [], [], [], [], []
    obs_list2, acts_list2, next_obs2 = [], [], []
    step_count = 0

    # opponent_specification = {
    #                     'entirely_self': (1.0, partial(BasedAgent)),
    #                     #'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
    #                 }
    env = WarehouseBrawl()

    # Reset environment
    obs, _ = env.reset()
    policy1 = BasedAgent()
    policy2 = BasedAgent()
    policy1.get_env_info(env)
    policy2.get_env_info(env)

    while step_count < num_steps:
        # Get actions from both policies
        action_1 = policy1.predict(obs[0])  # First agent
        action_2 = policy2.predict(obs[1])  # Second agent

        # Store observation-action pairs
        obs_list.append(obs[0])
        acts_list.append(action_1)
        obs_list2.append(obs[1])
        acts_list2.append(action_2)

        # Step the environment
        obs, _, done, _, info = env.step({0: action_1, 1: action_2})
        next_obs.append(obs[0])
        next_obs2.append(obs[1])
        dones.append(done)
        infos.append(info)
        
        step_count += 1  # Increment step count

        # Reset if any agent is done
        if done:
            obs, _ = env.reset()

    # Convert to Transitions format for imitation learning
    transitions = Transitions(
        obs=np.array(obs_list+obs_list2),  # Shape: (num_steps, obs_dim)
        acts=np.array(acts_list+acts_list2),  # Shape: (num_steps, num_agents)
        infos=np.array(infos+infos),
        next_obs=np.array(next_obs+next_obs2),  
        dones=np.array(dones+dones),
    )

    return transitions


def main(run_name: str='experiment_1', will_train=False, load: str=None, il_type: str=None, train_timesteps: int = 1_000_000, opponent_specification: dict = None, model_config=None):
    print(f"Running experiment: {run_name}")
    print(f"Will train: {will_train}")
    print(f"Load: {load}")
    print(f"il type: {il_type}")
    print(f"Train timesteps: {train_timesteps}")
    print(f"Opponent specification: {opponent_specification}")
    print(f"Model config: {model_config}")
    
    
    # Create agent
    # Start here if you want to train from scratch
    if not load:
        my_agent = SB3Agent(sb3_class=PPO, config=model_config)
    else:
        my_agent = SB3Agent(sb3_class=PPO, file_path=load, config=model_config)
    # Start here if you want to train from a specific timestep
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_180402_steps.zip')



    # Reward manager
    reward_manager = RewardManager(reward_functions, signal_subscriptions)

    if will_train:

        transitions = None
        if il_type:
            transitions = collect_transitions(num_steps=5000)

        # Self-play settings
        selfplay_handler = SelfPlayLatest(partial(SB3Agent, sb3_class=PPO))

        # Save settings
        save_handler = SaveHandler(
            agent=my_agent, # Agent to save
            save_freq=train_timesteps // 40, # Save frequency
            max_saved=-1, # Maximum number of saved models
            save_path='checkpoints', # Save path
            run_name=run_name,
            mode=SaveHandlerMode.RESUME # Save mode, FORCE or RESUME
        )

        # Opponent settings
        if not opponent_specification:
            opponent_specification = {
                            'self_play': (8, selfplay_handler),
                            'constant_agent': (2, partial(ConstantAgent)),
                        }
        opponent_cfg = OpponentsCfg(opponents=opponent_specification)
        print(f"Opponent specification: {opponent_specification}")

        train(my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_timesteps=train_timesteps,                 # plot doesn't work properly with small timesteps
            train_logging=TrainLogging.PLOT,
            il_type=il_type,
            transitions=transitions
        )

    test_opponent = BasedBetterAgent()
    # evaluate_against_pool(my_agent, test_opponent, num_matches=20)
    match_time = 90
    os.makedirs('replays', exist_ok=True)
    run_match(my_agent,
            agent_2=test_opponent,
            video_path=f'replays/{run_name}.mp4',
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            max_timesteps=30 * match_time
            )


if __name__ == '__main__':
    main(will_train=True, il_type='bc', run_name='experiment_bc_based', train_timesteps=0)
    # run_match(SB3Agent(file_path='checkpoints/experiment_bc_dodge_3/rl_model_0_steps.zip'), BasedAgent(), video_path='dodge_bc.mp4')
    # run_match(BasedAgent(), BasedAgent(), video_path='based_based.mp4')
