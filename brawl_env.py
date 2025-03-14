from typing import Any, Callable, List, Tuple
from enum import Enum, auto
from dataclasses import dataclass

import os, random, json
import numpy as np
import gymnasium
import pygame
import pygame.gfxdraw
import pymunk
import pymunk.pygame_util

from Malachite import AgentID, MalachiteEnv
from camera import Camera, CameraResolution, RenderMode
from game_object import GameObject, Ground, Player
from low_high_class import ActHelper, ObsHelper
from reward_utils import determine_win, euclidean_distance, get_hitting_damage, is_hitting, opponent_side_from_agent

class Signal():
    def __init__(self, env):
        self._handlers: List[Callable] = []
        self.env = env

    def connect(self, handler: Callable):
        self._handlers.append(handler)

    def emit(self, *args, **kwargs):
        for handler in self._handlers:
            handler(self.env, *args, **kwargs)

class Result(Enum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"

@dataclass
class PlayerStats():
    damage_taken: float
    damage_taken_this_stock: float
    damage_done: float
    lives_left: int

@dataclass
class MatchStats():
    match_time: float  # Total match time in seconds
    player1: PlayerStats
    player2: PlayerStats
    player1_result: Result


# Define an enumeration for the moves
class MoveType(Enum):
    NONE = auto()         # no move
    NLIGHT = auto()       # grounded light neutral
    DLIGHT = auto()       # grounded light down
    SLIGHT = auto()       # grounded light side
    NSIG = auto()         # grounded heavy neutral
    DSIG = auto()         # grounded heavy down
    SSIG = auto()         # grounded heavy side
    NAIR = auto()         # aerial light neutral
    DAIR = auto()         # aerial light down
    SAIR = auto()         # aerial light side
    RECOVERY = auto()     # aerial heavy neutral and aerial heavy side
    GROUNDPOUND = auto()  # aerial heavy down

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)

# Define a frozen dataclass for the key
@dataclass(frozen=True)
class CompactMoveState():
    grounded: bool
    heavy: bool
    direction_type: int

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

class Facing(Enum):
    RIGHT = 1
    LEFT = -1

    def __int__(self):
        return self.value

    @staticmethod
    def flip(facing):
        return Facing.LEFT if facing == Facing.RIGHT else Facing.RIGHT

    @staticmethod
    def from_direction(direction: float) -> "Facing":
        return Facing.RIGHT if direction > 0 else Facing.LEFT

    @staticmethod
    def turn_check(facing, direction) -> bool:
        if facing == Facing.RIGHT and direction < 0:
            return True
        if facing == Facing.LEFT and direction > 0:
            return True
        return False

"""Coord system
    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y
"""

class WarehouseBrawl(MalachiteEnv[np.ndarray, np.ndarray, int]):

    BRAWL_TO_UNITS = 1.024 / 320  # Conversion factor

    def __init__(self, mode: RenderMode=RenderMode.RGB_ARRAY, resolution: CameraResolution=CameraResolution.LOW, train_mode: bool = False):
        super(WarehouseBrawl, self).__init__()

        self.stage_width_tiles: float = 29.8
        self.stage_height_tiles: float = 16.8

        self.mode = mode
        self.resolution = resolution
        self.train_mode = train_mode

        self.agents = [0, 1] # Agent 0, agent 1
        self.logger = ['', '']

        # Params
        self.fps = 30
        self.dt = 1 / self.fps
        self.max_timesteps = self.fps * 90

        self.agent_1_name = 'Team 1'
        self.agent_2_name = 'Team 2'

        # Signals
        self.knockout_signal = Signal(self)
        self.win_signal = Signal(self)
        self.hit_during_stun = Signal(self)

        # Observation Space
        self.observation_space = self.get_observation_space()

        self.camera = Camera()

        # Action Space
        # WASD
        self.action_space = self.get_action_space()
        # spaces.Box(low=np.array([0] * 4), high=np.array([1] * 4), shape=(4,), dtype=np.float32)

        self.action_spaces, self.observation_spaces = {}, {}
        for agent_id in self.agents:
            self.action_spaces[agent_id] = self.action_space
            self.observation_spaces[agent_id] = self.observation_space

        self.load_attacks()

        self.reset()

    def get_observation_space(self):
        # lowarray = np.array(
        #     [0, -self.screen_width_tiles/2, -self.screen_width_tiles/2, 0, 0, 0, 0, 0] +
        #     [0 for _ in range(len(Player.states))] +
        #     [0] +
        #     [(0, -self.screen_width_tiles, -self.screen_width_tiles, 0, 0)[i%5] for i in range(self.max_ammo*5)] +
        #     [0, -self.screen_width_tiles/2, -self.screen_width_tiles/2, 0, -self.screen_width_tiles, -self.screen_width_tiles, -self.screen_width_tiles, -self.screen_width_tiles,
        #     0, 0, 0, 0] +
        #     [0 for _ in range(len(Player.states))] +
        #     [(0, -self.screen_width_tiles, -self.screen_width_tiles, 0, 0)[i%5] for i in range(self.max_ammo*5)] +
        #     [0]
        # )
        # higharray = np.array(
        #     [1, self.screen_width_tiles/2, self.screen_width_tiles/2, self.screen_width_tiles/2, 2 * math.pi, 10, 20, 3] +
        #     [1 for _ in range(len(Player.states))] +
        #     [2*math.pi] +
        #     [(1, self.screen_width_tiles, self.screen_width_tiles, 2*math.pi, 2*math.pi)[i%5] for i in range(self.max_ammo*5)] +
        #     [1, self.screen_width_tiles/2, self.screen_width_tiles/2, self.screen_width_tiles/2, self.screen_width_tiles, self.screen_width_tiles, self.screen_width_tiles, self.screen_width_tiles,
        #     2 * math.pi, 2 * math.pi, 20, 3] +
        #     [1 for _ in range(len(Player.states))] +
        #     [(1, self.screen_width_tiles, self.screen_width_tiles, 2*math.pi, 2*math.pi)[i%5] for i in range(self.max_ammo*5)] +
        #     [self.time_limit]
        # )

        obs_helper = ObsHelper()
        self.add_player_obs(obs_helper, 'player')
        self.add_player_obs(obs_helper, 'opponent')

        print('Obs space', obs_helper.low, obs_helper.high)

        self.obs_helper = obs_helper

        return self.obs_helper.get_as_box()

    def add_player_obs(self, obs_helper, name: str='player') -> None:
        # Note: Some low and high bounds are off here. To ensure everyone's code
        # still works, we are not modifying them, but will elaborate in comments.
        # Pos: Unnormalized, goes from [-18, -7], [18, 7], in game units
        obs_helper.add_section([-1, -1], [1, 1], f"{name}_pos")
        # Vel: Unnormalized, goes from [-10, -10], [10, 10] in game units
        obs_helper.add_section([-1, -1], [1, 1], f"{name}_vel")
        obs_helper.add_section([0], [1], f"{name}_facing")
        obs_helper.add_section([0], [1], f"{name}_grounded")
        obs_helper.add_section([0], [1], f"{name}_aerial")
        obs_helper.add_section([0], [2], f"{name}_jumps_left")
        obs_helper.add_section([0], [12], f"{name}_state")
        obs_helper.add_section([0], [1], f"{name}_recoveries_left")
        # Dodge timer: Unnormalized, goes from [0], [82] in frames.
        # Represents the time remaining until can dodge again
        obs_helper.add_section([0], [1], f"{name}_dodge_timer")
        # Stun frames: Unnormalized, goes from [0], [80] in frames
        # Represents the time remaining until the player transitions
        # out of StunState.
        obs_helper.add_section([0], [1], f"{name}_stun_frames")
        obs_helper.add_section([0], [1], f"{name}_damage")
        obs_helper.add_section([0], [3], f"{name}_stocks")
        obs_helper.add_section([0], [11], f"{name}_move_type")

    def get_action_space(self):
        act_helper = ActHelper()
        act_helper.add_key("w") # W (Aim up)
        act_helper.add_key("a") # A (Left)
        act_helper.add_key("s") # S (Aim down/fastfall)
        act_helper.add_key("d") # D (Right)
        act_helper.add_key("space") # Space (Jump)
        act_helper.add_key("h") # H (Pickup/Throw)
        act_helper.add_key("l") # L (Dash/Dodge)
        act_helper.add_key("j") # J (Light Attack)
        act_helper.add_key("k") # K (Heavy Attack)
        act_helper.add_key("g") # G (Taunt)

        print('Action space', act_helper.low, act_helper.high)

        self.act_helper = act_helper

        return self.act_helper.get_as_box()

    def square_floor_collision(arbiter, space, data):
        """
        Collision handler callback that is called when a square collides with the platform.
        It sets the square's collision flag so that is_on_floor() returns True.
        """
        shape_a, shape_b = arbiter.shapes
        # Check both shapes; one of them should be a square.
        if hasattr(shape_a, "owner") and isinstance(shape_a.owner, Player):
            shape_a.owner.collided_this_step = True
        if hasattr(shape_b, "owner") and isinstance(shape_b.owner, Player):
            shape_b.owner.collided_this_step = True
        return True

    def get_stats(self, agent_id: int) -> PlayerStats:
        player = self.players[agent_id]
        return PlayerStats(
            damage_taken=player.damage_taken_total,
            damage_taken_this_stock=player.damage_taken_this_stock,
            damage_done=player.damage_done,
            lives_left=player.stocks)

    def load_attacks(self):
        # load all from /content/attacks
        self.attacks = {}

        self.keys = {
            'Unarmed NLight': MoveType.NLIGHT,
            'Unarmed DLight': MoveType.DLIGHT,
            'Unarmed SLight': MoveType.SLIGHT,
            'Unarmed NSig':   MoveType.NSIG,
            'Unarmed DSig':   MoveType.DSIG,
            'Unarmed SSig':   MoveType.SSIG,
            'Unarmed NAir':   MoveType.NAIR,
            'Unarmed DAir':   MoveType.DAIR,
            'Unarmed SAir':   MoveType.SAIR,
            'Unarmed Recovery': MoveType.RECOVERY,
            'Unarmed Groundpound': MoveType.GROUNDPOUND,
        }

        for file in sorted(os.listdir('attacks')):
            name = file.split('.')[0]
            if name not in self.keys.keys(): continue
            with open(os.path.join('attacks', file)) as f:
                move_data = json.load(f)

            self.attacks[self.keys[name]] = move_data


    def step(self, action: dict[int, np.ndarray]):
        # Create new rewards dict
        self.cur_action = action
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminated = False
        self.logger = ['', '']

        self.camera.process()

        # Process all other steps
        for obj_name, obj in self.objects.items():
            # If player
            if isinstance(obj, Player):
                continue
            else:
                obj.process()
        # Pre-process player step
        for agent in self.agents:
            player = self.players[agent]
            player.pre_process()

        # Process player step
        for agent in self.agents:
            player = self.players[agent]
            player.process(action[agent])
            if player.stocks <= 0:
                self.terminated = True
                self.win_signal.emit(agent='player' if agent == 1 else 'opponent')


        # Process physics info
        for obj_name, obj in self.objects.items():
            obj.physics_process(self.dt)

        # PyMunk step
        self.space.step(self.dt)
        self.steps += 1

        truncated = self.steps >= self.max_timesteps

        if truncated:
            player1_result = determine_win(self.get_stats(0), self.get_stats(1))
            if player1_result == Result.WIN:
                self.win_signal.emit(agent='player')
            elif player1_result == Result.LOSS:
                self.win_signal.emit(agent='opponent')

        # Collect observations
        observations = {agent: self.observe(agent) for agent in self.agents}

        return observations, self.rewards, self.terminated, truncated, {}

    def add_reward(self, agent: int, reward: float) -> None:
        # Not really in use
        self.rewards[agent] += reward

    def reset(self, seed=None) -> Tuple[dict[int, np.ndarray], dict[str, Any]]:
        self.seed = seed

        self.space = pymunk.Space()
        self.dt = 1 / 30.0
        self.space.gravity = 0, 17.808

        self.steps = 0

        # Other params
        self.rewards = {agent: 0 for agent in self.agents}

        # Game Objects
        self.objects: dict[str, GameObject] = {}

        self.players: list[Player] = []
        self.camera.reset(self)
        self._setup()

        return {agent: self.observe(agent) for agent in self.agents}, {}

    def observe(self, agent: int) -> np.ndarray:
        #  lh = LowHigh()
        # lh += [-1, -1], [1, 1] # 2d vector to goal
        # lh += [-1, -1], [1, 1] # 2d vector of global position
        # lh += [-1, -1], [1, 1] # 2d vector of global velocity

        obs = []
        obs += self.players[agent].get_obs()
        obs += self.players[1-agent].get_obs()
        #obs += self.players[agent].body.position.x, self.players[agent].body.position.y
        #obs += self.players[agent].body.position.x, self.players[agent].body.position.y
        #obs += self.players[agent].body.velocity.x, self.players[agent].body.velocity.y

        return np.array(obs)

    def render(self) -> None | np.ndarray | str | list:
        return self.camera.get_frame(self)

    def handle_ui(self, canvas: pygame.Surface) -> None:
        # Define UI
        # player_stat = f"P1: {self.players[0].stocks}, P2: {self.players[1].stocks}"
        # text_surface = self.camera.font.render(player_stat, True, (255, 255, 255))  # White text
        # text_rect = text_surface.get_rect(center=(self.camera.window_width // 2, 50))  # Center the text
        # canvas.blit(text_surface, text_rect)

        # # Damage
        # small_font = pygame.font.Font(None, 20)
        # text_surface = small_font.render(f"{self.players[0].damage}%, {self.players[1].damage}%", True, (255, 255, 255))  # White text
        # text_rect = text_surface.get_rect(center=(self.camera.window_width // 2, 70))  # Center the text
        # canvas.blit(text_surface, text_rect)

        # Smaller text
        small_font = pygame.font.Font(None, 30)
        text_surface = small_font.render(f"Time: {self.steps}", True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(self.camera.window_width // 2, 30))  # Center the text
        canvas.blit(text_surface, text_rect)

        # Smaller text
        small_font = pygame.font.Font(None, 20)
        text_surface = small_font.render(f"P1: {self.logger[0]['transition']}, P2: {self.logger[1]['transition']}", True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(self.camera.window_width // 2, 50))  # Center the text
        canvas.blit(text_surface, text_rect)

        # Smaller text
        small_font = pygame.font.Font(None, 20)
        text_surface = small_font.render(f"P1: {self.logger[0].get('move_type', '')}, P2: {self.logger[1].get('move_type', '')}", True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(self.camera.window_width // 2, 70))  # Center the text
        canvas.blit(text_surface, text_rect)

        # Smaller text
        text_surface = small_font.render(f"P1 Total Reward: {self.logger[0].get('total_reward', '')}, Reward {self.logger[0].get('reward', '')}", True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 40))  # Center the text
        # make it left
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)

        text_surface = small_font.render(f"P2 Total Reward: {self.logger[1].get('total_reward', '')}, Reward {self.logger[1].get('reward', '')}", True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 20))  # Center the text
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)

        for idx, (key, val) in enumerate(self.logger[0].get('reward_breakdown', {}).items()):
            color = (255, 255, 255)    # White text
            if val > 0:
                color = (0, 255, 0)
            elif val < 0:
                color = (255, 0, 0)
            text_surface = small_font.render(f"{key}: {val:.8f}", True, color) 
            text_rect = text_surface.get_rect(center=(0, self.camera.window_height -40 - 20*idx))  # Center the text
            text_rect.right = self.camera.window_width - 10 
            canvas.blit(text_surface, text_rect)

        # Print player positions
        player_pos = self.players[0].body.position
        opponent_pos = self.players[1].body.position

        player_vel = self.players[0].body.velocity
        opponent_vel = self.players[1].body.velocity

        # Define platform boundaries
        platform_center_x = 0.0
        platform_width = 10.67

        left_edge = platform_center_x - (platform_width / 2)
        right_edge = platform_center_x + (platform_width / 2)

        platform_height = 1.54 
        
        color = [255, 255, 255]
        if player_pos.x < left_edge or player_pos.x > right_edge:
            color[2] -= 255
        if player_pos.y > platform_height + 0.2:
            color[1] -= 255

        facing = 'Right' if self.players[0].facing == Facing.RIGHT else 'Left'
        attacking = 'Attacking' if is_hitting(self.players[0]) else ''
        attacking_power = get_hitting_damage(self.players[0]) if attacking else ''
        shielding = 'Shielding' if self.players[0].state.vulnerable() == False else ''

        text_surface = small_font.render(f"P1 Pos: ({player_pos.x:.2f}, {player_pos.y:.2f}) Velocity: ({player_vel.x:.2f}, {player_vel.y:.2f}) Facing: {facing} {attacking} {attacking_power} {shielding}", True, color)  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 60))  # Center the text
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)

        color = [255, 255, 255]
        if opponent_pos.x < left_edge or opponent_pos.x > right_edge:
            color[2] -= 255
        if opponent_pos.y > platform_height + 0.2:
            color[1] -= 255

        facing = 'Right' if self.players[1].facing == Facing.RIGHT else 'Left'
        attacking = 'Attacking' if is_hitting(self.players[1]) else ''
        attacking_power = get_hitting_damage(self.players[1]) if attacking else ''
        shielding = 'Shielding' if self.players[1].state.vulnerable() == False else ''

        text_surface = small_font.render(f"P2 Pos: ({opponent_pos.x:.2f}, {opponent_pos.y:.2f}) Velocity: ({opponent_vel.x:.2f}, {opponent_vel.y:.2f}) Facing: {facing} {attacking} {attacking_power} {shielding}", True, color)  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 80))  # Center the text
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)

        eclid_dist = euclidean_distance(self.players[0], self.players[1])
        opponent_side_from_player = 'Right' if opponent_side_from_agent(self.players[0], self.players[1]) == Facing.RIGHT else 'Left'
        
        text_surface = small_font.render(f"Distance: {eclid_dist:.2f} Opponent: {opponent_side_from_player}", True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 100))  # Center the text
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)

        
        jump_timer = self.players[0].state.jump_timer if hasattr(self.players[0].state, 'jump_timer') else None
        jumps_left = self.players[0].state.jumps_left if hasattr(self.players[0].state, 'jumps_left') else None
        text_surface = small_font.render(f"P1 can_jump: {jump_timer} jumps_left: ({jumps_left})", True, color)  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 120))  # Center the text
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)

        jump_timer = self.players[1].state.jump_timer if hasattr(self.players[1].state, 'jump_timer') else None
        jumps_left = self.players[1].state.jumps_left if hasattr(self.players[1].state, 'jumps_left') else None
        text_surface = small_font.render(f"P2 can_jump: {jump_timer} jumps_left: ({jumps_left})", True, color)  # White text
        text_rect = text_surface.get_rect(center=(0, self.camera.window_height - 140))  # Center the text
        text_rect.left = 0
        canvas.blit(text_surface, text_rect)



    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def close(self) -> None:
        self.camera.close()

    def _setup(self):
        # Collsion fix
        handler = self.space.add_collision_handler(3, 4)  # (Player1 collision_type, Player2 collision_type)
        handler.begin = lambda *args, **kwargs: False

        # Environment
        ground = Ground(self.space, 0, 2.03, 10.67)
        self.objects['ground'] = ground

        # Players
        # randomize start pos, binary
        p1_right = bool(random.getrandbits(1))
        p1_start_pos = [5, 0] if p1_right else [-5, 0]
        p2_start_pos = [-5, 0] if p1_right else [5, 0]

        # Uncomment this if you'd like. It makes train_mode RANDOMIZE the
        # position of both players, so that they get used to many
        # different positions in the map!

        if self.train_mode:
            print('Randomizing start position')
            p1_start_pos = [random.uniform(-5, 5), 0]
            p2_start_pos = [random.uniform(-5, 5), 0]
        else:
            p1_start_pos = [5, 0] if p1_right else [-5, 0]
            p2_start_pos = [-5, 0] if p1_right else [5, 0]

        p1 = Player(self, 0, start_position=p1_start_pos, color=[0, 0, 255, 255])
        p2 = Player(self, 1, start_position=p2_start_pos, color=[0, 255, 0, 255])

        self.objects['player'] = p1
        self.objects['opponent'] = p2

        self.players += [p1, p2]
