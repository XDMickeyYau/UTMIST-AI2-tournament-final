"""
Main goals:
1. Get back to the platform
2. Hit/Stun the enemy
3. Stay alive

Getback to the platform 
1. Continuously penalize slightly for downwards velocity
2. Significant reward for inwards and upwards velocity
3. (Optional) Reward for approaching the center of the platform more than our enemy

Hit/stun the enemy
1. Sparse reward for hitting the enemy
2. Higher reward for hitting when the enemy gets stunned

Stay alive
1. Reward for dodging the enemy's attacks
2. A very low continuous reward for moving back and forth.
2.5 (Optional) Reward for high velocity. 
3. (Optional, keep it constant for the moment) Penalize slightly for getting hit by the enemy 

"""

"""Reward functions with getting back to the platform"""
from enum import Enum
from typing import Type

import numpy as np
from brawl_env import Facing, MoveType, WarehouseBrawl
from game_object import Player, GameObject
from reward_manager import RewTerm, RewardManager
from player import AttackState, Cast, GroundState, InAirState, KOState, PlayerObjectState, BackDashState, Power, StunState, TauntState
from reward_utils import euclidean_distance, get_hitting_damage, is_hitting, left_edge, opponent_side_from_agent, right_edge, platform_height, cushion



# --- Reward functions ---
class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")


    return reward / 140 

def damage_interaction_bonus(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
    multiplier: float = 0.5,
) -> float:
    
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    reward = damage_interaction_reward(env, mode)

    if reward > 0:
        if hasattr(opponent.state, "can_control") and not opponent.state.can_control():
            return reward * multiplier
    elif reward < 0:
        if hasattr(player.state, "can_control") and not player.state.can_control():
            return reward* multiplier
    elif opponent.facing == opponent_side_from_agent(player, opponent):
        return reward* multiplier

    return 0

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * np.clip(player.body.velocity.x, -10, 10)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Do nothing if the opponent is knocked out
    if isinstance(opponent.state, KOState):
        return 0

    # Set the opponent's position within the platform boundaries to prevent the player from running out of the edge
    opponent_position_x = opponent.body.position.x
    if opponent_position_x < left_edge:
        opponent_position_x = left_edge
    elif opponent_position_x > right_edge:
        opponent_position_x = right_edge

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent_position_x else 1
    reward = multiplier * np.clip(player.body.velocity.x, -10, 10)

    return reward

def head_to_opponent_jump(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Do nothing if the opponent is knocked out
    if isinstance(opponent.state, KOState):
        return 0
    
    opponent_position_y = opponent.body.position.y
    if opponent_position_y > platform_height:
        opponent_position_y = platform_height

    if player.body.position.y == opponent_position_y:
        return 0
    
    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.y > opponent_position_y else 1
    reward = multiplier * np.clip(player.body.velocity.y, -10, 10)

    return reward


def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def return_platform_reward_horizontal(
    env: WarehouseBrawl,
) -> float:
    # Get player object from the environment
    player: Player = env.objects["player"]

    if player.is_on_floor():
        return 0
    
    if left_edge < player.body.position.x < right_edge:
        return 0

    # Set the opponent's position within the platform boundaries to prevent the player from running out of the edge
    opponent_position_x = None
    if player.body.position.x < left_edge:
        opponent_position_x = left_edge
    elif player.body.position.x > right_edge:
        opponent_position_x = right_edge

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent_position_x else 1
    reward = multiplier * np.clip(player.body.velocity.x, -10, 10)

    return reward


def return_platform_reward_vertical(
    env: WarehouseBrawl,
) -> float:
    # Get player object from the environment
    player: Player = env.objects["player"]
    
    if player.is_on_floor():
        return 0
    
    if player.body.position.y < platform_height:
        return 0

    opponent_position_y = None
    if player.body.position.y > platform_height:
        opponent_position_y = platform_height

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.y > opponent_position_y else 1
    reward = multiplier * np.clip(player.body.velocity.y, -10, 10)

    return reward


    
reward_functions = {
# ----- Getback to the platform -----
"return_platform_reward_horizontal": RewTerm(func=return_platform_reward_horizontal,weight=0.000_05),
"return_platform_reward_vertical": RewTerm(func=return_platform_reward_vertical,weight=0.000_025),

# ----- Stay Alive -----
'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.000_05),
'head_to_opponent_jump': RewTerm(func=head_to_opponent_jump, weight=0.000_025),
'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.000_025),


# ----- Hit/Stun the Enemy -----
'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=3.0),
'damage_interaction_bonus': RewTerm(func=damage_interaction_bonus, weight=3.0),
}

signal_subscriptions = {
    'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
    'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
}
