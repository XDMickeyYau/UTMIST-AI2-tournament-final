"""
1. prioritize recoveries 
2. is enemy attacking and in distance? then dodge
3. if not approach enemy and hit them 



2. prioritize floatiness 
3. if distance > threshold and enemy is wasting move, punish 
4. else, just focus on dodging 

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

import random
from typing import Optional
import numpy as np

from agents import Agent

class DodgeBasedAgent(Agent):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        self.last_opponent_pos = None
        self.safe_distance = 2.0  # Increased safe distance
        self.aggressive_range = 1.0  # Range for aggressive attacks
        self.combo_range = 1.5  # Range for combo attempts
        self.edge_guard_target = None  # target for edge guarding
        self.falling_off_stage = False  # flag for when falling
        self.offstage_recovery_timer = 0  # timer for recovery attempts
        self.time = 0
        self.move_time = 0
        self.moves_queue = 'InAirState'
        self.tried_dodging = False

        self.last_state = None

    def predict(self, obs):
        # Action Definitions (Initialize)
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

        # Combo Actions (Example - Adapt these based on the game)
        LIGHT_ATTACK_RIGHT = self.act_helper.press_keys(
            ['d', 'j'])  # Move Right + Light Attack
        LIGHT_ATTACK_LEFT = self.act_helper.press_keys(
            ['a', 'j'])  # Move Left + Light Attack
        LIGHT_ATTACK_DOWN = self.act_helper.press_keys(
            ['s', 'j'])
        LIGHT_ATTACK_NEUTRAL = self.act_helper.press_keys(
            ['w', 'j'])
        HEAVY_ATTACK_RIGHT = self.act_helper.press_keys(
            ['d', 'k'])  # Move Right + Heavy Attack
        HEAVY_ATTACK_LEFT = self.act_helper.press_keys(
            ['a', 'k'])  # Move Left + Heavy Attack
        HEAVY_ATTACK_DOWN = self.act_helper.press_keys(
            ['s', 'k'])
        HEAVY_ATTACK_NEUTRAL = self.act_helper.press_keys(
            ['w', 'k'])
        RECOVERY = self.act_helper.press_keys(['w', 'k'])
        GROUNDPOUND = self.act_helper.press_keys(['s', 'k'])

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

        distance_x = abs(opponent_pos[0] - player_pos[0])
        distance_y = abs(opponent_pos[1] - player_pos[1])
        distance = np.sqrt(distance_x**2 + distance_y**2)

        self.time += 1

        """
        MODES: 
        - OnPlatform
        - Recovery
        """

        action = None
        turned_around = False

        if player_state == state_mapping['StunState']:
            return IDLE

        if abs(player_pos[0]) < 10.7 / 2:
            mode = 'OnPlatform'
        else:
            mode = 'Recovery'
        if distance < 2.0:
            if not self.tried_dodging:
                self.tried_dodging = True
                mode = 'Dodge'
            else:
                self.tried_dodging = False
                mode = 'Attack'
        

        # OnPlatform mode
        if mode == 'OnPlatform':
            if self.time >= self.move_time or player_state == state_mapping['WalkingState']:
                action = JUMP
                self.move_time = self.time + 12

        # Recovery mode
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
        elif mode == 'Attack' \
                and player_dodge_timer <= 0 \
                and player_state != state_mapping['AttackState']\
                and player_state != state_mapping['StunState']\
                and player_pos[0] < 5.0 \
                and player_pos[0] > -5.0:
            options = []
            if player_grounded:
                if player_pos[0] < opponent_pos[0]:
                    options = [LIGHT_ATTACK_DOWN,LIGHT_ATTACK_NEUTRAL, LIGHT_ATTACK_RIGHT, HEAVY_ATTACK_DOWN, HEAVY_ATTACK_NEUTRAL, HEAVY_ATTACK_RIGHT]
                else:
                    options = [LIGHT_ATTACK_DOWN,LIGHT_ATTACK_NEUTRAL, LIGHT_ATTACK_LEFT, HEAVY_ATTACK_DOWN, HEAVY_ATTACK_NEUTRAL, HEAVY_ATTACK_LEFT]
            elif player_aerial:
                if player_pos[0] < opponent_pos[0]:
                    options = [LIGHT_ATTACK_DOWN,LIGHT_ATTACK_NEUTRAL, LIGHT_ATTACK_RIGHT, GROUNDPOUND]
                else:
                    options = [LIGHT_ATTACK_DOWN,LIGHT_ATTACK_NEUTRAL, LIGHT_ATTACK_LEFT, GROUNDPOUND]
            action = random.choice(options)


        if action is None:
            if player_pos[0] < 0:
                action = RIGHT
            else:
                action = LEFT

        # if player_state == state_mapping['StunState']:
        #     print(player_jumps_left)

        return action
