from functools import partial
import os
import shutil

import numpy as np
from stable_baselines3 import PPO
from DodgeBasedAgent import DodgeBasedAgent
from agents import BasedBetterAgent, ConstantAgent, RandomAgent, RecurrentPPOAgent, SB3Agent, BasedAgent, SB3AgentRuled
from brawl_env import WarehouseBrawl
from camera import CameraResolution
from evaluation import evaluate_against_pool
from reward import reward_functions, signal_subscriptions
from reward_manager import RewardManager
from run_agent import TrainLogging, run_match, train
from self_play import SelfPlayDynamic, SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode, SelfPlayHandler
from imitation.data.types import Transitions
from imitation.data import rollout
from main import main

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


if __name__ == '__main__':

    import argparse

    # ----------------- Parse arguments -----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_steps", type=int, default=0, help="Number of steps to load from pretrained model.")
    parser.add_argument("--train_steps", type=int, default=1_000_000, help="Number of timesteps to train.")
    parser.add_argument("--version", type=str, default="", help="Version of the model.")
    parser.add_argument("--ent_coef", type=float, default=0., help="Entropy coefficient.")
    parser.add_argument("--il", type=str, default="", help="Imitation learning.", choices=["", "il", "dodge","betterbased"])
    parser.add_argument("--self_play_mode", type=str, default="latest", help="Self-play selection mode")
    parser.add_argument("--agent_class", type=str, default="SB3Agent", help="Agent class")
    args = parser.parse_args()

    # ----------------- Print arguments -----------------
    pretrained_steps = args.pretrained_steps
    train_steps = args.train_steps
    version = args.version
    ent_coef = args.ent_coef
    il = args.il
    self_play_mode = args.self_play_mode
    agent_class_str = args.agent_class

    print(f"Pretrained steps: {pretrained_steps}")
    print(f"Train steps: {train_steps}")
    print(f"Version: {version}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"IL: {il}")
    print(f"Self-play mode: {self_play_mode}")
    print(f"Agent class: {agent_class_str}")

    # ----------------- Choose the agent class -----------------
    if agent_class_str == 'SB3Agent':
        agent_class = partial(SB3Agent,sb3_class=PPO)
    elif agent_class_str == 'SB3AgentRuled':
        agent_class = partial(SB3AgentRuled,sb3_class=PPO)
    else:
        raise ValueError(f"Unknown agent class: {agent_class_str}")

    print(f"Agent class: {agent_class}")
    
    # ----------------- Choose the model to run -----------------
    chosen_name = f'ver{version}ppo_{(train_steps+pretrained_steps) // 1000}k'
    if il:
        chosen_name += f'_{il}'
    if ent_coef:
        chosen_name += f'_ent_coef_{ent_coef}'
    if self_play_mode:
        chosen_name += f'_{self_play_mode}'
    if agent_class_str == 'SB3AgentRuled':
        chosen_name += f'_ruled'
    
    print(f"Chosen name: {chosen_name}")

    # ----------------- Create new checkpoints form pretrained -----------------
    if pretrained_steps:
        pretrained_name = f'ver{version}ppo_{(pretrained_steps) // 1000}k'
        if il:
            pretrained_name += f'_{il}'
        if ent_coef:
            pretrained_name += f'_ent_coef_{ent_coef}'
        if self_play_mode:
            pretrained_name += f'_{self_play_mode}'
        if agent_class_str == 'SB3AgentRuled':
            pretrained_name += f'_ruled'
    
        print(f"Pretrained name: {pretrained_name}")

        shutil.copytree(f'checkpoints/{pretrained_name}', f'checkpoints/{chosen_name}')

        print(f"Copied checkpoints from {pretrained_name} to {chosen_name}")
    

    # ----------------- Choose the checkpoint to run -----------------
    # Define the checkpoint paths in a dictionary
    checkpoint_paths = {
        0: {
            "betterbased": 'checkpoints/experiment_better_based/rl_model_0_steps.zip',
            "dodge": 'checkpoints/experiment_bc_dodge_3/rl_model_0_steps.zip',
            "il" : 'checkpoints/experiment_5/rl_model_2048_steps.zip',
            "": None
        },
        1_000_000: {
            "betterbased": f'checkpoints/{chosen_name}/rl_model_1004400_steps.zip',
            "dodge": f"checkpoints/{chosen_name}/rl_model_1004400_steps.zip",
            "il": f'checkpoints/{chosen_name}/rl_model_1006448_steps.zip',
            "": f'checkpoints/{chosen_name}/rl_model_1001472_steps.zip'
        },
        2_000_000: {
            "betterbased": f'checkpoints/{chosen_name}/rl_model_2008800_steps.zip',
            "il": f'checkpoints/{chosen_name}/rl_model_2010848_steps.zip',
            "": f'checkpoints/{chosen_name}/rl_model_2005872_steps.zip'
        },
        3_000_000: {
            "il": f'checkpoints/{chosen_name}/rl_model_3015248_steps.zip',
            "": f'checkpoints/{chosen_name}/rl_model_3010272_steps.zip'
        }
    }

    # Get the checkpoint path based on pretrained_steps and il
    if pretrained_steps in checkpoint_paths:
        checkpoints_path = checkpoint_paths[pretrained_steps][il]
    else:
        raise ValueError(f"Unknown pretrained steps: {pretrained_steps}")

    print(f"Checkpoints path: {checkpoints_path}")
    # ----------------- Choose the opponent to run -----------------
     # Self-play settings
    # selfplay_handler = SelfPlayHandler(
    #     agent_class, # Agent class and its keyword arguments
    #     mode=SelfPlaySelectionMode.LATEST # Self-play selection mode
    # )

    if self_play_mode == 'latest':
        opponent_specification = None
    elif self_play_mode == "based_mixed":
        self_play_handler = SelfPlayDynamic(partial(SB3Agent, sb3_class=PPO), random_prob=0.5)
        opponent_specification = {
                        'self_play': (6.6, self_play_handler),
                        'base_better_agent': (3.3, partial(BasedBetterAgent)),
        }
    elif self_play_mode == "4_mixed":
                self_play_handler = SelfPlayDynamic(partial(SB3Agent, sb3_class=PPO), random_prob=0.33)
                opponent_specification = {
                        'self_play': (6, self_play_handler),
                        'base_better_agent': (2, partial(BasedBetterAgent)),
                        'dodge_based_agent': (2, partial(DodgeBasedAgent)),
                        'random_agent': (1, partial(RandomAgent)),
                        'constant_agent': (1, partial(ConstantAgent)),
        }
    else:
        raise ValueError(f"Unknown self-play mode: {self_play_mode}")
    
    print(f"Opponent specification: {opponent_specification}")

    # ----------------- Choose the model config -----------------
    model_config = {}
    if ent_coef:
        model_config['ent_coef'] = ent_coef

    # ----------------- Run the experiment -----------------
    # model_config_str = '_'.join([f'{k}_{v}' for k, v in model_config.items()])
    main(run_name=chosen_name, 
         will_train=True, 
         load=checkpoints_path, 
         train_timesteps=train_steps, 
         model_config=model_config, 
         opponent_specification=opponent_specification)
    # run_match(SB3Agent(file_path='checkpoints/experiment_5/rl_model_2048_steps.zip'), ConstantAgent(), video_path='based.mp4')
