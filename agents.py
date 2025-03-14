
import warnings
from typing import TYPE_CHECKING, Any, Generic, \
    SupportsFloat, TypeVar, Type, Optional, List, Dict, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from functools import partial
from typing import Tuple, Any

from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

import gdown, os, math, random, shutil, json

import numpy as np
import torch
from torch import nn

import gymnasium
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
import pygame

from agents_rules import get_facing, get_obs, prevent_silly_actions, stay_alive, state_mapping, enhance_attacking
from brawl_env import Facing


SelfAgent = TypeVar("SelfAgent", bound="Agent")

class Agent(ABC):

    def __init__(
            self,
            file_path: Optional[str] = None
        ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env):
        if isinstance(env, Monitor):
            self_env = env.env
        else:
            self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return

class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action
    
class UserInputAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)
        return action

class ClockworkAgent(Agent):

    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
            ]
        else:
            self.action_sheet = action_sheet


    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)


        self.steps += 1  # Increment step counter
        return action

class ClockworkAgentRuled(ClockworkAgent):
    
    def __init__(self, action_sheet = None):
        super().__init__(action_sheet)
        self.move_time = 0
        # self.move_time = self.time + 15
    
    
    def jump(self, action):
        self.act_helper.press_keys(['space'], action)
        self.move_time = self.steps + 15
    
    def dodge(self, action):
        self.act_helper.press_keys(['l'], action)
        action[self.act_helper.sections['space']] = 0
    
    def get_back_to_platform(self, obs, action):
        pos = self.obs_helper.get_section(obs, 'player_pos')
        vel = self.obs_helper.get_section(obs, 'player_vel')
        facing = get_facing(self.obs_helper, obs)
        grounded = get_obs(self.obs_helper, obs, 'player_grounded')
        dodge_timer = get_obs(self.obs_helper, obs, 'player_dodge_timer')
        jump_left = get_obs(self.obs_helper, obs, 'player_jumps_left')
        recovery_left = get_obs(self.obs_helper, obs, 'player_recoveries_left')
        state = get_obs(self.obs_helper, obs, 'player_state')
        ground_level = 1.54
        speed_threshold = 3

        # If grounded, return action
        if grounded:
            return action

        # If on the platform, return action
        if abs(pos[0]) <= 10.67/2:
            return action
        
        # Check if the player is turning around
        turned_around = False
        if (pos[0] < 0 and vel[0] > 0 and facing == Facing.RIGHT) or (pos[0] > 0 and vel[0] < 0 and facing == Facing.LEFT):
            turned_around = True
        
        # Just do it every 2 frames
        if self.steps % 3:
            return action

        # If cannot control, do nothing
        if state in [state_mapping['StunState'], state_mapping['KOState'], state_mapping['TauntState'], state_mapping['DodgeState'], state_mapping['AttackState']]:
            return action
        
        can_jump = self.steps > self.move_time and jump_left > 0
        can_dodge = dodge_timer <= 0
        can_recovery = recovery_left > 0 
        
        if pos[1] > ground_level and vel[1] > 0 and can_jump: # TESTED
            self.jump(action)
        elif pos[1] > ground_level and vel[1] > 0 and can_dodge: # TESTED
            self.dodge(action)
        elif pos[0] < - 10.67/2 and vel[0] < -speed_threshold and facing == Facing.RIGHT and can_dodge:
            self.dodge(action)
        elif pos[0] > 10.67/2 and vel[0] > speed_threshold  and facing == Facing.LEFT and can_dodge:
            self.dodge(action)
        elif turned_around and can_jump:
            self.jump(action)
        
        return action


    
    def predict(self, obs):
        action = super().predict(obs)
        action = enhance_attacking(self.obs_helper, obs, self.act_helper, action)
        # action = prevent_silly_actions(self.obs_helper, obs, self.act_helper, action)
        # action = self.get_back_to_platform(obs, action)
        # action = stay_alive(self.obs_helper, obs, self.act_helper, action, self.steps)
        return action

#%%
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

class SB3Agent(Agent):

    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        self.sb3_class = sb3_class
        self.config = config if config is not None else {}
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, **self.config)         # GAIL: n_steps=16384
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, **self.config) #official: ent_coef=0.01
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, n_steps=30*90*3, batch_size=128, **self.config)
            # official: self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,


        )

class SB3AgentRuled(SB3Agent):

    def __init__(self, sb3_class = PPO, file_path = None, config = None):
        super().__init__(sb3_class, file_path, config)
        self.time = 0
        self.move_time = 0
        # self.move_time = self.time + 15
    
    
    def jump(self, action):
        self.act_helper.press_keys(['space'], action)
        self.move_time = self.time + 15
    
    def dodge(self, action):
        self.act_helper.press_keys(['l'], action)
        action[self.act_helper.sections['space']] = 0
    
    def get_back_to_platform(self, obs, action):
        pos = self.obs_helper.get_section(obs, 'player_pos')
        vel = self.obs_helper.get_section(obs, 'player_vel')
        facing = get_facing(self.obs_helper, obs)
        grounded = get_obs(self.obs_helper, obs, 'player_grounded')
        dodge_timer = get_obs(self.obs_helper, obs, 'player_dodge_timer')
        jump_left = get_obs(self.obs_helper, obs, 'player_jumps_left')
        recovery_left = get_obs(self.obs_helper, obs, 'player_recoveries_left')
        state = get_obs(self.obs_helper, obs, 'player_state')
        ground_level = 1.54
        speed_threshold = 3

        # If grounded, return action
        if grounded:
            return action

        # If on the platform, return action
        if abs(pos[0]) <= 10.67/2:
            return action
        
        # Check if the player is turning around
        turned_around = False
        if (pos[0] < 0 and vel[0] > 0 and facing == Facing.RIGHT) or (pos[0] > 0 and vel[0] < 0 and facing == Facing.LEFT):
            turned_around = True
        
        # Just do it every 2 frames
        if self.time % 3:
            return action
        
        # If cannot control, do nothing
        if state in [state_mapping['StunState'], state_mapping['KOState'], state_mapping['TauntState'], state_mapping['DodgeState'], state_mapping['AttackState']]:
            return action
        
        can_jump = self.time > self.move_time and jump_left > 0
        can_dodge = dodge_timer <= 0
        can_recovery = recovery_left > 0 
        
        if pos[1] > ground_level and vel[1] > 0 and can_jump: # TESTED
            self.jump(action)
        elif pos[1] > ground_level and vel[1] > 0 and can_dodge: # TESTED
            self.dodge(action)
        elif pos[0] < - 10.67/2 and vel[0] < -speed_threshold and facing == Facing.RIGHT and can_dodge:
            self.dodge(action)
        elif pos[0] > 10.67/2 and vel[0] > speed_threshold  and facing == Facing.LEFT and can_dodge: # TESTED
            self.dodge(action)
        elif turned_around and can_jump: # TESTED
            self.jump(action)
        
        return action


    
    def predict(self, obs):
        self.time += 1
        action, _ = self.model.predict(obs)
        action = enhance_attacking(self.obs_helper, obs, self.act_helper, action)
        action = prevent_silly_actions(self.obs_helper, obs, self.act_helper, action)
        action = self.get_back_to_platform(obs, action)
        action = stay_alive(self.obs_helper, obs, self.act_helper, action, self.time)
        return action

#%%
class A2CAgent(Agent):

    def __init__(
        self,
        sb3_class: Optional[Type[BaseAlgorithm]] = A2C,
        file_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.sb3_class = sb3_class
        self.config = config if config is not None else {}
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, **self.config)
            del self.env
        else:
            load_config = self.config().copy()
            load_config.pop("batch_size", None) # A2C does not use batch size
            self.model = self.sb3_class.load(
                self.file_path,
                n_steps = 30 * 90 * 3,
                **load_config
            )
    
    def _gdown(self) -> str:
        return ""

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent(Agent):

    def __init__(
            self,
            file_path: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.config = config if config is not None else {}

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = RecurrentPPO("MlpLstmPolicy", self.env, verbose=0, **self.config)
            # official: self.model = RecurrentPPO("MlpLstmPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path, n_steps=30*90*3, batch_size=128, **self.config)
            # official: self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 16, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class SubmittedAgent(Agent):

    def __init__(
            self,
            file_path: Optional[str] = None,
            # example_argument = 0,
    ):
        # Your code here
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            print('hii')
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path)
            # self.model = A2C.load(self.file_path)
            # self.model = SAC.load(self.file_path)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1G60ilYtohdmXsYyjBtwdzC1PRBerqpfJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):

    def __init__(
            self,
            file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 3 == 0:
            action = self.act_helper.press_keys(['space'], action)
        
        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            random_attack = random.choice(['j', 'k', 'l'])
            action = self.act_helper.press_keys([random_attack], action)
            direction = random.choice(['w', 'w', 's', 's', 'a', 'd'])
            action = self.act_helper.press_keys([direction], action)
            if direction == 's':
                action[self.act_helper.sections['a']] = 0
                action[self.act_helper.sections['d']] = 0
        return action
    
class BasedBetterAgent(BasedAgent):

    def __init__(self, file_path = None):
        super().__init__(file_path)
        self.move_time = 0
        # self.move_time = self.time + 15
    
    
    def jump(self, action):
        self.act_helper.press_keys(['space'], action)
        self.move_time = self.time + 15
    
    def dodge(self, action):
        self.act_helper.press_keys(['l'], action)
        action[self.act_helper.sections['space']] = 0
    
    def get_back_to_platform(self, obs, action):
        pos = self.obs_helper.get_section(obs, 'player_pos')
        vel = self.obs_helper.get_section(obs, 'player_vel')
        facing = get_facing(self.obs_helper, obs)
        grounded = get_obs(self.obs_helper, obs, 'player_grounded')
        dodge_timer = get_obs(self.obs_helper, obs, 'player_dodge_timer')
        jump_left = get_obs(self.obs_helper, obs, 'player_jumps_left')
        recovery_left = get_obs(self.obs_helper, obs, 'player_recoveries_left')
        state = get_obs(self.obs_helper, obs, 'player_state')
        ground_level = 1.54
        speed_threshold = 3

        # If grounded, return action
        if grounded:
            return action

        # If on the platform, return action
        if abs(pos[0]) <= 10.67/2:
            return action
        
        # Check if the player is turning around
        turned_around = False
        if (pos[0] < 0 and vel[0] > 0 and facing == Facing.RIGHT) or (pos[0] > 0 and vel[0] < 0 and facing == Facing.LEFT):
            turned_around = True
        
        # Just do it every 2 frames
        if self.time % 3:
            return action
        
        # If cannot control, do nothing
        if state in [state_mapping['StunState'], state_mapping['KOState'], state_mapping['TauntState'], state_mapping['DodgeState'], state_mapping['AttackState']]:
            return action
        
        can_jump = self.time > self.move_time and jump_left > 0
        can_dodge = dodge_timer <= 0
        can_recovery = recovery_left > 0 
        
        if pos[1] > ground_level and vel[1] > 0 and can_jump: # TESTED
            self.jump(action)
        elif pos[1] > ground_level and vel[1] > 0 and can_dodge: # TESTED
            self.dodge(action)
        elif pos[0] < - 10.67/2 and vel[0] < -speed_threshold and facing == Facing.RIGHT and can_dodge:
            self.dodge(action)
        elif pos[0] > 10.67/2 and vel[0] > speed_threshold  and facing == Facing.LEFT and can_dodge: # TESTED
            self.dodge(action)
        elif turned_around and can_jump: # TESTED
            self.jump(action)
        
        return action


    
    def predict(self, obs):
        action = super().predict(obs)
        action = prevent_silly_actions(self.obs_helper, obs, self.act_helper, action)
        action = self.get_back_to_platform(obs, action)
        action = stay_alive(self.obs_helper, obs, self.act_helper, action, self.time)
        return action

class ConstantEdgeJumpAgent(Agent):
    def __init__(self, file_path = None, **kwargs):
        super().__init__(file_path)
        self.margin = kwargs.get("margin", 2.0)
        self.time = 0
        self.move_time = 0
    
    def predict(self, obs):
        self.time += 1
        IDLE = self.act_helper.zeros()
        JUMP = self.act_helper.press_keys(['space'])
        RIGHT = self.act_helper.press_keys(['d'])
        LEFT = self.act_helper.press_keys(['a'])
        
        player_pos = self.obs_helper.get_section(obs, 'player_pos')
        if 10.67/2 - self.margin <= player_pos[0] or player_pos[0] <= -10.67/2 + self.margin:
            # Cooldown
            if self.time >= self.move_time:
                self.move_time = self.time + 6
                return JUMP
            else:
                return IDLE

        elif player_pos[0] > 0:
            return RIGHT
        
        return LEFT