import numpy as np
from brawl_env import CompactMoveState, Facing, MoveType
from low_high_class import ActHelper, ObsHelper

# ----------------- Mapping -----------------

# def add_player_obs(self, obs_helper, name: str='player') -> None:
#     obs_helper.add_section([-1, -1], [1, 1], f"{name}_pos")
#     obs_helper.add_section([-1, -1], [1, 1], f"{name}_vel")
#     obs_helper.add_section([0], [1], f"{name}_facing")
#     obs_helper.add_section([0], [1], f"{name}_grounded")
#     obs_helper.add_section([0], [1], f"{name}_aerial")
#     obs_helper.add_section([0], [2], f"{name}_jumps_left")
#     obs_helper.add_section([0], [12], f"{name}_state")
#     obs_helper.add_section([0], [1], f"{name}_recoveries_left")
#     obs_helper.add_section([0], [1], f"{name}_dodge_timer")
#     obs_helper.add_section([0], [1], f"{name}_stun_frames")
#     obs_helper.add_section([0], [1], f"{name}_damage")
#     obs_helper.add_section([0], [3], f"{name}_stocks")
#     obs_helper.add_section([0], [11], f"{name}_move_type")

# def get_action_space(self):
#     act_helper = ActHelper()
#     act_helper.add_key("w") # W (Aim up)
#     act_helper.add_key("a") # A (Left)
#     act_helper.add_key("s") # S (Aim down/fastfall)
#     act_helper.add_key("d") # D (Right)
#     act_helper.add_key("space") # Space (Jump)
#     act_helper.add_key("h") # H (Pickup/Throw)
#     act_helper.add_key("l") # L (Dash/Dodge)
#     act_helper.add_key("j") # J (Light Attack)
#     act_helper.add_key("k") # K (Heavy Attack)
#     act_helper.add_key("g") # G (Taunt)

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

# ----------------- Helper Functions -----------------

def is_facing_near_edge(pos, facing: Facing, threshold: float=2.5) -> bool:
    return (pos[0] < -threshold and facing == Facing.LEFT) or (pos[0] > threshold and facing == Facing.RIGHT)

def get_facing(obs_helper: ObsHelper, obs: np.ndarray, opp: bool = False) -> Facing:
    facing = obs_helper.get_section(obs, 'player_facing' if not opp else 'opponent_facing')
    return Facing.RIGHT if facing == 1.0 else Facing.LEFT

def get_obs(obs_helper: ObsHelper, obs: np.ndarray, name: str) -> np.ndarray:
    return obs_helper.get_section(obs, name)[0]

def is_ground_state(state: int) -> bool:
    return state in [state_mapping['WalkingState'], state_mapping['StandingState'], state_mapping['TurnaroundState'] ,
                     state_mapping['SprintingState'], state_mapping['DashState'], state_mapping['BackDashState']]

def is_above_line(player_pos, point1=(-5.0, -1.54), point2=(-2.5, -2.67)) -> bool:
    # player_pos is a tuple or array (x, y)
    x, y = player_pos
    # Define the two points of the line
    x1, y1 = point1
    x2, y2 = point2
    # Compute slope
    m = (y2 - y1) / (x2 - x1)  # slope = -1.13 / 2.5
    # Compute the line's y at player's x
    y_line = y1 + m * (x - x1)
    # If player's y is below the line (i.e. smaller value), return True.
    # print(f"Player is above the line. pos: {y}, y_line: {y_line}")
    return y < y_line

def is_pressed(action: np.ndarray, key: str, act_helper: ActHelper) -> bool:
    return action[act_helper.sections[key]] > 0.5

# Create the dictionary mapping CompactMoveState to a Move
m_state_to_move = {
    CompactMoveState(True, False, 0): MoveType.NLIGHT,      # grounded light neutral
    CompactMoveState(True, False, 1): MoveType.DLIGHT,      # grounded light down
    CompactMoveState(True, False, 2): MoveType.SLIGHT,      # grounded light side
    CompactMoveState(True, True, 0): MoveType.NSIG,          # grounded heavy neutral
    CompactMoveState(True, True, 1): MoveType.DSIG,          # grounded heavy down
    CompactMoveState(True, True, 2): MoveType.SSIG,          # grounded heavy side
    CompactMoveState(False, False, 0): MoveType.NAIR,        # aerial light neutral
    CompactMoveState(False, False, 1): MoveType.DAIR,        # aerial light down
    CompactMoveState(False, False, 2): MoveType.SAIR,        # aerial light side
    CompactMoveState(False, True, 0): MoveType.RECOVERY,     # aerial heavy neutral
    CompactMoveState(False, True, 1): MoveType.GROUNDPOUND,  # aerial heavy down
    CompactMoveState(False, True, 2): MoveType.RECOVERY,     # aerial heavy side
}

def get_move(action: np.ndarray, act_helper: ActHelper, grounded: bool) -> MoveType:
    heavy_move = is_pressed(action, 'k', act_helper)         # heavy move if key 'k' is held
    light_move = (not heavy_move) and is_pressed(action, 'j', act_helper)  # light move if not heavy and key 'j' is held

    # Determine directional keys:
    left_key = is_pressed(action, "a", act_helper)            # left key (A)
    right_key = is_pressed(action, "d", act_helper)           # right key (D)
    up_key = is_pressed(action, "w", act_helper)              # aim up (W)
    down_key = is_pressed(action, "s", act_helper)            # aim down (S)

    # Calculate combined directions:
    side_key = left_key or right_key

    # Calculate move direction:
    neutral_move = ((not side_key) and (not down_key)) or up_key
    down_move = (not neutral_move) and down_key
    side_move = (not neutral_move) and (not down_key) and side_key

    # Check if any move key (light, heavy, or throw) is pressed:
    hitting_any_move_key = light_move or heavy_move
    if not hitting_any_move_key:
        move_type = MoveType.NONE
    else:
        # (Optional) Print the results:
        # print("heavy_move:", heavy_move)
        # print("light_move:", light_move)
        # print("throw_move:", throw_move)
        # print("neutral_move:", neutral_move)
        # print("down_move:", down_move)
        # print("side_move:", side_move)
        # print("hitting_any_move_key:", hitting_any_move_key)
        cms = CompactMoveState(grounded, heavy_move, 0 if neutral_move else (1 if down_move else 2))
        move_type = m_state_to_move[cms]
        #print(move_type)
    return move_type

# ----------------- Agent Rules -----------------
def prevent_SLight(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, threshold=2.5):
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    facing = get_facing(obs_helper, obs)
    grounded = get_obs(obs_helper, obs, 'player_grounded')

    # If the player is not grounded, return the action as is.
    if not grounded:
        return action

    # If the player is near the left edge and is facing left or vice versa, disable side attack.
    if not is_facing_near_edge(pos, facing, threshold):
        return action
    

    if is_pressed(action, 'j', act_helper):
        action[act_helper.sections['a']] = 0
        action[act_helper.sections['d']] = 0
        
    return action

def prevent_DAir(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, threshold=2.5):
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    facing = get_facing(obs_helper, obs)
    grounded = get_obs(obs_helper, obs, 'player_grounded')

    # Define the line that separates the stage from the edge.
    max_height = -2.67
    ground_level = 1.54
    edge_size = 5

    # If the player is grounded, return the action as is.
    if grounded:
        return action

    if not is_facing_near_edge(pos, facing, threshold):
        return action
    
    # If the player is moving left and is facing right or vice versa, convert down air attack to groundpound
    if facing == Facing.LEFT and is_above_line(pos,(-threshold, max_height),(-edge_size, ground_level)) \
        or facing == Facing.RIGHT and is_above_line(pos,(threshold, max_height),(edge_size, ground_level)):
        if is_pressed(action, 'j', act_helper) and is_pressed(action, 's', act_helper):
            action[act_helper.sections['j']] = 0
            action[act_helper.sections['k']] = 1
        
    return action
    
def prevent_dash(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, threshold=2) -> np.ndarray:
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    facing = get_facing(obs_helper, obs)
    grounded = get_obs(obs_helper, obs, 'player_grounded')
    state = get_obs(obs_helper, obs, 'player_state')

    # If the player is not grounded, return the action as is.
    if not grounded:
        return action

    # If the player is near the left edge and is facing left or vice versa, disable dash.
    if not is_facing_near_edge(pos, facing, threshold):
        return action
    
    # If not WalkingState or SprintingState, return the action as is.
    if state not in [state_mapping['WalkingState'], state_mapping['SprintingState']]:
        return action

    if facing == Facing.LEFT and pos[0] < -threshold:
        if not is_pressed(action, 'a', act_helper) and is_pressed(action, 'd', act_helper):
            return action
        else:
            action[act_helper.sections['l']] = 0
    elif facing == Facing.RIGHT and pos[0] > threshold:
        if is_pressed(action, 'a', act_helper) and not is_pressed(action, 'd', act_helper):
            return action
        else:
            action[act_helper.sections['l']] = 0
        
    return action

def prevent_moving_out_of_edge(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, threshold=5) -> np.ndarray:
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    facing = get_facing(obs_helper, obs)
    grounded = get_obs(obs_helper, obs, 'player_grounded')

    # print(f"Player is near the edge. pos: {pos}, facing: {facing}, grounded: {grounded}")

    # If the player is not grounded, return the action as is.
    if not grounded:
        return action

    # If the player is near the left edge and is facing left or vice versa, disable dash.
    if not is_facing_near_edge(pos, facing, threshold):
        return action

    # If the player is moving left and is facing right or vice versa, disable movement.
    if facing == Facing.LEFT and pos[0] < -threshold:
        action[act_helper.sections['a']] = 0
        # print("Player is near the left edge. Disabling left movement.")
    elif facing == Facing.RIGHT and pos[0] > threshold:
        action[act_helper.sections['d']] = 0
        # print("Player is near the right edge. Disabling right movement.")

    return action


def prevent_picking_and_taunting(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray) -> np.ndarray:
    # Get the opponent's state
    opp_state = get_obs(obs_helper, obs, 'opponent_state')

    # No picking action
    action[act_helper.sections['h']] = 0

    # No tauant action if the opponent is not in KOState
    if opp_state != state_mapping['KOState']:
        action[act_helper.sections['g']] = 0

    return action

# ----------------- Staying Alive -----------------

def fall_when_above_edge(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, height=4) -> np.ndarray:
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    vel = obs_helper.get_section(obs, 'player_vel')
    grounded = get_obs(obs_helper, obs, 'player_grounded')

    if grounded:
        return action
    

    if pos[1] < -height and vel[1] < 0:
        action[act_helper.sections['s']] = 1
        action[act_helper.sections['k']] = 1

    return action

def keep_turning_toward_platform(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray) -> np.ndarray:
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    grounded = get_obs(obs_helper, obs, 'player_grounded')

    if grounded:
        return action

    # If the player is moving left and is facing right or vice versa, disable movement.
    if pos[0] < - 10.67/2:
        action[act_helper.sections['d']] = 1
    elif pos[0] > 10.67/2:
        action[act_helper.sections['a']] = 1

    return action

def prevent_attacking_when_leaving(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray) -> np.ndarray:
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    vel = obs_helper.get_section(obs, 'player_vel')
    grounded = get_obs(obs_helper, obs, 'player_grounded')
    ground_level = 1.54

    if grounded:
        return action

    # If the player is moving left and is facing right or vice versa, disable movement.
    if pos[0] < - 10.67/2: 
        action[act_helper.sections['j']] = 0
        action[act_helper.sections['k']] = 0
    elif pos[0] > 10.67/2:
        action[act_helper.sections['j']] = 0
        action[act_helper.sections['k']] = 0
    if pos[1] > ground_level:
        action[act_helper.sections['j']] = 0
        action[act_helper.sections['k']] = 0

    return action

# ----------------- Enhance attacking capabilities -----------------
def attack_opp_enemy(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, dsig_range: float = 1.0, turn_range: float = 2.0) -> np.ndarray:
    # Get the player's position and facing direction.
    pos = obs_helper.get_section(obs, 'player_pos')
    facing = get_facing(obs_helper, obs)
    opp_facing = get_facing(obs_helper, obs, opp=True)
    opp_pos = obs_helper.get_section(obs, 'opponent_pos')
    own_state = get_obs(obs_helper, obs, 'player_state')
    grounded = get_obs(obs_helper, obs, 'player_grounded')
    dist = ((pos[0] - opp_pos[0]) ** 2 +(pos[1] - opp_pos[1]) ** 2) ** 0.5

    move = get_move(action, act_helper, bool(grounded))

    # If no move key is pressed, return the action as is.
    # if move == MoveType.NONE:
    #     return action
    
    # If side light to flee, return the action as is.
    if move == MoveType.SLIGHT:
        return action
    
    # If down signiture to attack both side, return the action as is.
    if move == MoveType.DSIG:
        return action
    
    if own_state == state_mapping['StandingState'] and ((facing == opp_facing == Facing.LEFT and pos[0] < opp_pos[0]) or (facing == opp_facing == Facing.RIGHT and pos[0] > opp_pos[0])):
        if dist < turn_range:
            # Face Opponent
            action[act_helper.sections['j']] = 0
            action[act_helper.sections['k']] = 0
            if facing == Facing.LEFT:
                action[act_helper.sections['a']] = 0
                action[act_helper.sections['d']] = 1
            elif facing == Facing.RIGHT:
                action[act_helper.sections['a']] = 1
                action[act_helper.sections['d']] = 0

            if dist < dsig_range:
                # Jump (and possibly ground pound if opponent can't move)
                action[act_helper.sections['space']] = 1

    return action

def attack_more_ground_pound(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, own_vel_range: float=7.0, opp_vel_range: float=1.0, attack_x_range: float=0.75, edge_threshold: float=10.67/2) -> np.ndarray:
    own_state = get_obs(obs_helper, obs, 'player_state')
    own_vel = obs_helper.get_section(obs, 'player_vel')
    opp_vel = obs_helper.get_section(obs, 'opponent_vel')
    own_pos = obs_helper.get_section(obs, 'player_pos')
    opp_pos = obs_helper.get_section(obs, 'opponent_pos')
    opp_state = get_obs(obs_helper, obs, 'opponent_state')
    opp_grounded = get_obs(obs_helper, obs, 'opponent_grounded')

    non_controllable_states = [state_mapping['TauntState'], state_mapping['StunState'], state_mapping['KOState'], state_mapping['AttackState']]
    air_states = [state_mapping['InAirState'], state_mapping['AirTurnaroundState']]

    if own_vel[0] < own_vel_range and opp_grounded and opp_vel[0] < opp_vel_range and opp_state in non_controllable_states and own_state in air_states:
        pos_to_attack_x = own_pos[0] + own_vel[0] / 3       # assuming 30 fps and 10 frames to charge
        # print(f"pos_to_attack_x: {pos_to_attack_x}, opp_pos[0]: {opp_pos[0]}")
        if abs(pos_to_attack_x) < edge_threshold and abs(pos_to_attack_x - opp_pos[0]) < attack_x_range:
            # Ground Pound
            action[act_helper.sections['k']] = 1
            action[act_helper.sections['w']] = 0
            action[act_helper.sections['s']] = 1

    return action

# ----------------- Applying All Rules -----------------
def prevent_silly_actions(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray) -> np.ndarray:
    action = prevent_SLight(obs_helper, obs, act_helper, action) # TESTED
    action = prevent_DAir(obs_helper, obs, act_helper, action) # TESTED
    action = prevent_dash(obs_helper, obs, act_helper, action) # TESTED
    action = prevent_picking_and_taunting(obs_helper, obs, act_helper, action) # TESTED
    action = prevent_moving_out_of_edge(obs_helper, obs, act_helper, action) # TESTED
    return action

def stay_alive(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray, time) -> np.ndarray:
    action = fall_when_above_edge(obs_helper, obs, act_helper, action) # TESTED
    action = keep_turning_toward_platform(obs_helper, obs, act_helper, action) # TESTED
    action = prevent_attacking_when_leaving(obs_helper, obs, act_helper, action) # TESTED
    return action

def enhance_attacking(obs_helper: ObsHelper, obs: np.ndarray, act_helper: ActHelper, action: np.ndarray) -> np.ndarray:
    action = attack_opp_enemy(obs_helper, obs, act_helper, action) # TESTED
    action = attack_more_ground_pound(obs_helper, obs, act_helper, action) # TESTED
    return action