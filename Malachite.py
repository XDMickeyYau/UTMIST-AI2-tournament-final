from typing import Any, Generic, TypeVar, Dict
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import gymnasium


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
AgentID = TypeVar("AgentID")

# Reference PettingZoo AECEnv
class MalachiteEnv(ABC, Generic[ObsType, ActType, AgentID]):

    agents: list[AgentID]

    action_spaces: dict[AgentID, gymnasium.spaces.Space]
    observation_spaces: Dict[
        AgentID, gymnasium.spaces.Space
    ]

    # Whether each agent has just reached a terminal state
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    rewards: dict[AgentID, float]  # Reward from the last step for each agent
    # Cumulative rewards for each agent
    _cumulative_rewards: dict[AgentID, float]
    infos: dict[
        AgentID, dict[str, Any]
    ]  # Additional information from the last step for each agent

    def __init__(self):
        pass

    @abstractmethod
    def step(self, action: dict[AgentID, ActType]) -> tuple[ObsType,]:
        pass

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        pass

    @abstractmethod
    def observe(self, agent: AgentID) -> ObsType | None:
        pass

    @abstractmethod
    def render(self) -> None | np.ndarray | str | list:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def show_image(self, image: np.ndarray) -> None:
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]
