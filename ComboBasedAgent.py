import random
from typing import Optional, List
import numpy as np

from agents import Agent

class ComboBasedAgent(Agent):
    """
1. First, prioritize recoveries, head to the center of the platform
2. Second, prioritize to stick to the passed combos
3. Otherwise just dodge.

notes: 
- 2 jumps max, each 12 frames apart 
- 19 nlight
- 20 slight
- 20 dlight

STATES: 
- GroundState
- InAirState
- TauntState
- WalkingState
- SprintingState
- StandingState
- TurnaroundState
- AirTurnaroundState
- StunState
- KOState
- DashState
- BackDashState
- DodgeState
- AttackState

"""
    def __init__(self, file_path: Optional[str] = None, **kwargs): 
        super().__init__(file_path)
        
        self.margin = kwargs.get("margin", 0.5)
        self.attack_range = kwargs.get("attack_range", 2.0)
        self.combo_active = False
        self.combo_sequence = []  # Current combo being executed
        self.combo_index = 0
        
        self.left_end = -(10.7/2)
        self.right_end = 10.7/2
        
        self.time = 0
        self.move_time = 0
        self.tried_dodging = False
        
    def predict(self, obs):
        IDLE = self.act_helper.zeros()
        LEFT = self.act_helper.press_keys(['a'])
        RIGHT = self.act_helper.press_keys(['d'])
        UP = self.act_helper.press_keys(['w'])
        DOWN = self.act_helper.press_keys(['s'])
        JUMP = self.act_helper.press_keys(['space'])
        NEUTRAL_LIGHT = self.act_helper.press_keys(['j'])
        NEUTRAL_HEAVY = self.act_helper.press_keys(['k'])
        DODGE = self.act_helper.press_keys(['l'])
        DODGE_LEFT = self.act_helper.press_keys(['a', 'l'])
        DODGE_RIGHT = self.act_helper.press_keys(['d', 'l'])
        DODGE_UP = self.act_helper.press_keys(['w', 'l'])
        DODGE_DOWN = self.act_helper.press_keys(['s', 'l'])
        RECOVERY = self.act_helper.press_keys(['w', 'k'])

        # Define combo actions (predefined moves)
        self.LIGHT_ATTACK_DOWN = self.act_helper.press_keys(['s', 'j'])
        self.HEAVY_ATTACK_DOWN = self.act_helper.press_keys(['s', 'k'])
        self.LIGHT_ATTACK_NEUTRAL = self.act_helper.press_keys(['w', 'j'])
        self.HEAVY_ATTACK_NEUTRAL = self.act_helper.press_keys(['w', 'k'])
        self.LIGHT_ATTACK_RIGHT = self.act_helper.press_keys(['d', 'j'])
        self.HEAVY_ATTACK_RIGHT = self.act_helper.press_keys(['d', 'k'])
        self.LIGHT_ATTACK_LEFT = self.act_helper.press_keys(['a', 'j'])
        self.HEAVY_ATTACK_LEFT = self.act_helper.press_keys(['a', 'k'])
        
        # For rightward combos:
        self.combo_horizontal_right_neutral = [self.LIGHT_ATTACK_RIGHT, RIGHT, self.LIGHT_ATTACK_NEUTRAL]
        self.combo_horizontal_right_down = [self.LIGHT_ATTACK_RIGHT, RIGHT, self.LIGHT_ATTACK_DOWN]
        # For leftward combos:
        self.combo_horizontal_left_neutral = [self.LIGHT_ATTACK_LEFT, LEFT, self.LIGHT_ATTACK_NEUTRAL]
        self.combo_horizontal_left_down = [self.LIGHT_ATTACK_LEFT, LEFT, self.LIGHT_ATTACK_DOWN]
        # Vertical combos:
        self.combo_vertical_down_right = [self.LIGHT_ATTACK_DOWN, self.HEAVY_ATTACK_RIGHT]
        self.combo_vertical_down_left = [self.LIGHT_ATTACK_DOWN, self.HEAVY_ATTACK_LEFT]
        self.combo_vertical_down_neutral = [self.LIGHT_ATTACK_DOWN, self.HEAVY_ATTACK_NEUTRAL]
        self.combo_vertical_neutral_neutral = [self.LIGHT_ATTACK_NEUTRAL, self.HEAVY_ATTACK_NEUTRAL]
        self.combo_vertical_neutral_right = [self.LIGHT_ATTACK_NEUTRAL, self.LIGHT_ATTACK_RIGHT]
        self.combo_vertical_neutral_left = [self.LIGHT_ATTACK_NEUTRAL, self.LIGHT_ATTACK_LEFT]

        # Combine them into a list (not used directly now, since we choose via conditions)
        self.COMBOS = [
            self.combo_vertical_down_right,
            self.combo_vertical_down_left,
            self.combo_vertical_down_neutral,
            self.combo_vertical_neutral_neutral,
            self.combo_vertical_neutral_right,
            self.combo_vertical_neutral_left,
            self.combo_horizontal_right_neutral,
            self.combo_horizontal_right_down,
            self.combo_horizontal_left_neutral,
            self.combo_horizontal_left_down,
        ]

        self.time += 1
        
        action = self.act_helper.zeros()
        player_pos = self.obs_helper.get_section(obs, 'player_pos')
        player_vel = self.obs_helper.get_section(obs, 'player_vel')
        player_facing = self.obs_helper.get_section(obs, 'player_facing')
        player_grounded = self.obs_helper.get_section(obs, 'player_grounded')
        player_aerial = self.obs_helper.get_section(obs, 'player_aerial')
        player_jumps_left = self.obs_helper.get_section(
            obs, 'player_jumps_left')
        player_state = self.obs_helper.get_section(obs, 'player_state')
        player_recoveries_left = self.obs_helper.get_section(
            obs, 'player_recoveries_left')
        player_dodge_timer = self.obs_helper.get_section(
            obs, 'player_dodge_timer')
        player_stun_frames = self.obs_helper.get_section(
            obs, 'player_stun_frames')
        player_damage = self.obs_helper.get_section(obs, 'player_damage')
        player_stocks = self.obs_helper.get_section(obs, 'player_stocks')
        player_move_type = self.obs_helper.get_section(obs, 'player_move_type')

        opponent_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opponent_vel = self.obs_helper.get_section(obs, 'opponent_vel')
        opponent_facing = self.obs_helper.get_section(obs, 'opponent_facing')
        opponent_grounded = self.obs_helper.get_section(
            obs, 'opponent_grounded')
        opponent_aerial = self.obs_helper.get_section(obs, 'opponent_aerial')
        opponent_jumps_left = self.obs_helper.get_section(
            obs, 'opponent_jumps_left')
        opponent_state = self.obs_helper.get_section(obs, 'opponent_state')
        opponent_recoveries_left = self.obs_helper.get_section(
            obs, 'opponent_recoveries_left')
        opponent_dodge_timer = self.obs_helper.get_section(
            obs, 'opponent_dodge_timer')
        opponent_stun_frames = self.obs_helper.get_section(
            obs, 'opponent_stun_frames')
        opponent_damage = self.obs_helper.get_section(obs, 'opponent_damage')
        opponent_stocks = self.obs_helper.get_section(obs, 'opponent_stocks')
        opponent_move_type = self.obs_helper.get_section(
            obs, 'opponent_move_type')

        state_mapping = {
            'WalkingState': 0,
            'StandingState': 1,
            'TurnaroundState': 2,
            'AirTurnaroundState': 3,
            'SprintingState': 4,
            'StunState': 5,
            'InAirState': 6,
            'DodgeState': 7,
            'AttackState': 8,
            'DashState': 9,
            'BackDashState': 10,
            'KOState': 11,
            'TauntState': 12,
        }

        
        distance_x = opponent_pos[0] - player_pos[0]
        distance_y = opponent_pos[1] - player_pos[1]
        distance = np.sqrt(abs(distance_x)**2 + abs(distance_y)**2)
        
        if (distance < self.attack_range and 
            self.left_end + self.margin < player_pos[0] < self.right_end - self.margin):
            
            if distance_x > 0:
                # Opponent is to the right.
                # If opponent is moving slowly to the right, choose neutral combo; otherwise, use down combo.
                if opponent_vel[0] < 0.5:
                    self.combo_sequence = self.combo_horizontal_right_neutral
                else:
                    self.combo_sequence = self.combo_horizontal_right_down
            elif distance_x < 0:
                # Opponent is to the left.
                if opponent_vel[0] > -0.5:
                    self.combo_sequence = self.combo_horizontal_left_neutral
                else:
                    self.combo_sequence = self.combo_horizontal_left_down
            else:
                if player_pos[1] < opponent_pos[1]:
                    self.combo_sequence = self.combo_vertical_down_neutral
                else:
                    self.combo_sequence = self.combo_vertical_neutral_neutral
            self.combo_active = True
            self.combo_index = 0
        
        if self.combo_active:
            if self.combo_index < len(self.combo_sequence):
                action = self.combo_sequence[self.combo_index]
                self.combo_index += 1
                
                if self.combo_index >= len(self.combo_sequence):
                    self.combo_active = False
                return action
                
            else:
                self.combo_active = False

        else:
            # Prioritize recovery if combo is not active.
            # Copy and pasted from DodgeBasedAgent.py
            action = None
            turned_around = False

            if player_state == state_mapping['StunState']:
                return IDLE

            if abs(player_pos[0]) < 10.7 / 2:
                mode = 'OnPlatform'
            else:
                mode = 'Recovery'

            if distance < self.attack_range:
                if not self.tried_dodging:
                    self.tried_dodging = True
                    mode = 'Dodge'
                else:
                    self.tried_dodging = False
            
            # OnPlatform mode
            if mode == 'OnPlatform':
                if abs(player_pos[0]) <= 1.0:
                    if distance_x > 0:
                        action = RIGHT
                    else:
                        action = LEFT
                    self.move_time = self.time + 6
                else:
                    if self.time >= self.move_time or player_state == state_mapping['WalkingState']:
                        action = JUMP
                        self.move_time = self.time + 12

            elif mode == 'Recovery':
                if player_pos[0] < 0 and player_vel[0] > 0 or player_pos[0] > 0 and player_vel[0] < 0:
                    turned_around = True

                if player_state != state_mapping['StunState'] and turned_around:
                    if self.time >= self.move_time:
                        action = JUMP
                        self.move_time = self.time + 6
                    elif player_recoveries_left > 0:
                        action = RECOVERY

            elif mode == 'Dodge' \
                    and player_dodge_timer == 0 \
                    and opponent_state == state_mapping['AttackState'] \
                    and player_pos[0] < 5.0 \
                    and player_pos[0] > -5.0:

                # ON THE LEFT OF OPPONENT
                if player_pos[0] < opponent_pos[0] and opponent_facing == 0:
                    action = DODGE_RIGHT
                # ON THE RIGHT OF OPPONENT
                else:
                    action = DODGE_LEFT
            
            if action is None:
                if player_pos[0] < 0:
                    action = RIGHT
                else:
                    action = LEFT
            return action

# Inference use cases
# if "__main__" == __name__:
#     combo_config = {
#         "margin": 1.0,
#         "attack_range": 2.0
#     }

#     my_agent = ComboBasedAgent(**combo_config)