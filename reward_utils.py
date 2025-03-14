
from game_object import Player
from player import AttackState, Cast, Power

# --- Define platform boundaries ---
platform_center_x = 0.0
platform_width = 10.67

left_edge = platform_center_x - (platform_width / 2)
right_edge = platform_center_x + (platform_width / 2)

cushion = 0.35 # To avoid oscillations
platform_height = 1.54


# --- Helper functions ---
def opponent_side_from_agent(agent: Player, opponent: Player) -> "Facing":
    from brawl_env import Facing
    """Determine the side of the opponent from the agent."""
    return Facing.LEFT if opponent.body.position.x < agent.body.position.x else Facing.RIGHT

def is_hitting(agent: Player) -> bool:
    """Check if the agent is hitting the opponent."""
    return len(agent.hitboxes_to_draw) > 0

def euclidean_distance(agent: Player, opponent: Player) -> float:
    """Compute the Euclidean distance between the agent and the opponent."""
    return ((agent.body.position.x - opponent.body.position.x)**2 + (agent.body.position.y - opponent.body.position.y)**2)**0.5

def get_hitting_damage(agent: Player) -> float:
    """Get the damage of the agent's current attack."""
    if not isinstance(agent.state, AttackState):
        return 0
    
    # Get the current power and cast
    current_power: Power = agent.state.move_manager.current_power
    current_cast: Cast = current_power.casts[current_power.cast_idx]
    
    # Get the damage to deal
    cast_damage = current_cast.base_damage
    if current_power.damage_over_life_of_hitbox:
        damage_to_deal = cast_damage / current_cast.attack_frames
    else:
        damage_to_deal = cast_damage
    return damage_to_deal


# --- Determine win or lose ---
def determine_win(player_1_stats, player_2_stats):
    from brawl_env import Result
    if player_1_stats.lives_left > player_2_stats.lives_left:
        player1_result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        player1_result = Result.LOSS
    else:
        if player_1_stats.damage_taken_this_stock < player_2_stats.damage_taken_this_stock:
            player1_result = Result.WIN
        elif player_1_stats.damage_taken_this_stock > player_2_stats.damage_taken_this_stock:
            player1_result = Result.LOSS
        else:
            player1_result = Result.DRAW
    return player1_result
