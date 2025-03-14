import numpy as np
import pygame
from agents import ConstantAgent
from brawl_env import WarehouseBrawl
from camera import CameraResolution
from reward_manager import RewardManager
from pyvirtualdisplay import Display

class HumanControlledWarehouseBrawl(WarehouseBrawl):
    def __init__(self, reward_manager: RewardManager=None, resolution: CameraResolution=CameraResolution.LOW):
        super().__init__(resolution)
        # self.opponent_agent = ConstantAgent()
        self.reward_manager = reward_manager

    def get_human_input(self) -> dict[int, np.ndarray]:
        keys = pygame.key.get_pressed()
        action = np.zeros(self.action_space.shape)

        if keys[pygame.K_w]:
            action[0] = 1
        if keys[pygame.K_a]:
            action[1] = 1
        if keys[pygame.K_s]:
            action[2] = 1
        if keys[pygame.K_d]:
            action[3] = 1
        if keys[pygame.K_SPACE]:
            action[4] = 1
        if keys[pygame.K_h]:
            action[5] = 1
        if keys[pygame.K_l]:
            action[6] = 1
        if keys[pygame.K_j]:
            action[7] = 1
        if keys[pygame.K_k]:
            action[8] = 1
        if keys[pygame.K_g]:
            action[9] = 1

        return {0: action, 1: np.zeros(self.action_space.shape)} # assume Constant opponent

    def run_human_play(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            human_action = self.get_human_input()
            _, _, terminated, truncated, _ = self.step(human_action)

            # Print the reward for the human-controlled side (agent 0)
            if self.reward_manager is not None:
                rewards = self.reward_manager.process(self, 1 / 30.0)
                print(f"Step reward for human: {rewards}")

            screen.fill((0, 0, 0))
            frame = self.render()
            if frame is not None:
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))

            self.handle_ui(screen)
            pygame.display.flip()
            clock.tick(self.fps)

            if terminated or truncated:
                self.reset()

        pygame.quit()


if __name__ == "__main__":
    from reward import reward_functions
    manager = RewardManager(reward_functions)

    display = Display(visible=0, size=(800, 600))
    display.start()

    game = HumanControlledWarehouseBrawl(manager)
    game.run_human_play()
    