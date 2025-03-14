import random
from typing import Optional
import numpy as np
from agents import Agent

class GroundPoundBasedAgent(Agent):
    def __init__(self, file_path: Optional[str] = None, **kwargs):
        super().__init__(file_path)
        self.chase_threshold = kwargs.get("chase_threshold", 3.0)
        self.cooldown = kwargs.get("cooldown", 12)
        self.last_groundpound_time = -np.inf
        self.time = 0
        
    def predict(self, obs):
        IDLE = self.act_helper.zeros()
        LEFT = self.act_helper.press_keys(['a'])
        RIGHT = self.act_helper.press_keys(['d'])
        GROUNDPOUND = self.act_helper.press_keys(['s', 'k'])
        
        self.time += 1
        player_pos = self.obs_helper.get_section(obs, 'player_pos')
        opponent_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        
        dx = opponent_pos[0] - player_pos[0]
        distance = abs(dx)
        
        if self.time - self.last_groundpound_time < self.cooldown:
            if dx > 0:
                return RIGHT
            
            elif dx < 0:
                return LEFT
            
            else:
                return IDLE
        
        if distance < self.chase_threshold:
            self.last_groundpound_time = self.time
            return GROUNDPOUND

        else:
            if dx > 0:
                return RIGHT
            
            elif dx < 0:
                return LEFT
            
            else:
                return IDLE