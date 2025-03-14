
from enum import Enum

import numpy as np
import pygame
import pygame.gfxdraw
import pymunk
import pymunk.pygame_util
from pymunk.pygame_util import DrawOptions
from playable import UIHandler, KeyIconPanel

class CameraResolution(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class RenderMode(Enum):
    NONE = 0
    RGB_ARRAY = 1
    PYGAME_WINDOW = 2

class Camera():
    screen_width_tiles: float = 29.8
    screen_height_tiles: float = 16.8
    pixels_per_tile: float = 43
    is_rendering: bool = False
    space: pymunk.Space
    pos: list[int] = [0,0]
    zoom: float = 2.0


    def reset(self, env):
        self.space = env.space
        self.objects = env.objects
        self.resolution = env.resolution
        self.resolutions = {
            CameraResolution.LOW: (480, 720),
            CameraResolution.MEDIUM: (720, 1280),
            CameraResolution.HIGH: (1080, 1920)
        }

        self.window_height, self.window_width = self.resolutions[self.resolution]

        # WIDTH HEIGHT in Pixels
        #screen_width_tiles: float = 29.8
        #screen_height_tiles: float = 16.8
        self.pixels_per_tile = self.window_width // self.screen_width_tiles

        #self.window_width = self.screen_width_tiles * self.pixels_per_tile
        #self.window_height = self.screen_height_tiles * self.pixels_per_tile
        self.steps = 0

    def scale_gtp(self) -> float:
        return self.pixels_per_tile * self.zoom

    def _setup_render(self, mode) -> None:
        pygame.init()

        self.ui_handler = UIHandler(self)

        self.key_panel_1 = KeyIconPanel(side="left", edge_percentage=0.22, width_percentage=0.12, height_percentage=0.08)
        self.key_panel_2 = KeyIconPanel(side="right", edge_percentage=0.78, width_percentage=0.12, height_percentage=0.08)

        if mode == RenderMode.PYGAME_WINDOW:
            pygame.display.set_caption("Env")
            self.canvas = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        # Define font
        self.font50 = pygame.font.Font(None, 50)  # Use the default font with size 50
        self.font = pygame.font.Font(None, 50)

    def process(self) -> None:
        self.steps += 1

    def ptg(self, x, y=None) -> tuple[int, int]:
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
            x, y = x
        elif isinstance(x, pymunk.Vec2d):
            x, y = x.x, x.y

        scale_cst = self.scale_gtp()
        new_x = -self.screen_width_tiles / 2 + int(x / scale_cst)
        new_y = self.screen_height_tiles / 2 - int(y / scale_cst)
        return new_x, new_y

    def gtp(self, x, y=None) -> tuple[float, float]:
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
            x, y = x
        elif isinstance(x, pymunk.Vec2d):
            x, y = x.x, x.y

        scale_cst = self.scale_gtp()
        new_x = self.window_width / 2 + (x - self.pos[0]) * scale_cst
        new_y = self.window_height / 2 + (y -self.pos[1]) * scale_cst

        #new_x = self.window_width / 2 + x * self.pixels_per_tile
        #new_y = self.window_height / 2 + y * self.pixels_per_tile
        return new_x, new_y

    def get_frame(self, env, mode=RenderMode.RGB_ARRAY, has_hitboxes=False):
        if not self.is_rendering:
            self._setup_render(mode)
            self.is_rendering = True


        # Expose the canvas for editing
        if mode == RenderMode.RGB_ARRAY:
            self.canvas = pygame.Surface((self.window_width, self.window_height))
        #canvas = pygame.display.set_mode((self.window_width, self.window_height))
        self.canvas.fill((0, 0, 0))

        # Transform PyMunk objects to have (0,0) at center, and such that units are appropriate
        #center_x = self.window_width // 2
        #center_y = self.window_height // 2
        #scale = self.pixels_per_tile
        #transform = pymunk.Transform.identity().translated(center_x, center_y).scaled(scale)

        #center_x = self.screen_width_tiles // 2 - self.pos[0]
        #center_y = self.screen_height_tiles // 2 - self.pos[1]
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        scale = self.pixels_per_tile * self.zoom
        transform = pymunk.Transform.identity().translated(center_x, center_y).scaled(scale).translated(self.pos[0], self.pos[1])
        #transform = pymunk.Transform.identity().scaled(scale).translated(center_x, center_y).scaled(self.zoom)
        draw_options = DrawOptions(self.canvas)
        draw_options.transform = transform

        # Draw PyMunk objects
        #self.space.debug_draw(draw_options)

        #print(self.env.space)
        for obj_name, obj in self.objects.items():
            obj.render(self.canvas, self)

        # Draw UI + Text
        env.handle_ui(self.canvas)

        self.ui_handler.render(self, env)

        if hasattr(env, 'cur_action'):
            self.key_panel_1.draw(self, env.cur_action[0])
            self.key_panel_2.draw(self, env.cur_action[1])

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

        if mode == RenderMode.PYGAME_WINDOW:
            pygame.display.flip()
            pygame.event.pump()
            #pygame.display.update()
            self.clock.tick(50)

        return img

    def close(self) -> None:
        pygame.quit()