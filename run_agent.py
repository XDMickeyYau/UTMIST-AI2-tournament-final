from enum import Enum
from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np
import skvideo
import skvideo.io
from stable_baselines3 import PPO
from DodgeBasedAgent import DodgeBasedAgent
from agents import Agent, BasedAgent, BasedBetterAgent, ClockworkAgent, ClockworkAgentRuled, ConstantAgent, SB3Agent, SB3AgentRuled
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable, Optional, Union
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from brawl_env import MatchStats, Result, WarehouseBrawl
from camera import CameraResolution
from reward_manager import RewardManager
from reward import reward_functions, signal_subscriptions
from reward_utils import determine_win
from self_play import SelfPlayWarehouseBrawl, SelfPlayHandler, OpponentsCfg, SaveHandler
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial.gail import GAIL

import cv2
from tqdm import tqdm

# Example function to add reward text to frame
def add_reward_to_frame(frame, idx, name, reward):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30+40*idx)
    font_scale = 0.6
    font_color = (255, 0, 0)
    line_type = 2

    cv2.putText(frame, f'{name}: {reward:.3f}', 
                position, 
                font, 
                font_scale,
                font_color,
                line_type)
    return frame

def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              resolution = CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None,
              train_mode=False
              ) -> MatchStats:
    # Initialize env
    env = WarehouseBrawl(resolution=resolution, train_mode=train_mode)
    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]

    if reward_manager is not None:
        reward_manager.reset()
        reward_manager.subscribe_signals(env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name

    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30'  # Frame rate
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336

    for _ in tqdm(range(max_timesteps), total=max_timesteps):
        # actions = {agent: agents[agent].predict(None) for agent in range(2)}

        # observations, rewards, terminations, truncations, infos

        full_action = {
            0: agent_1.predict(obs_1),
            1: agent_2.predict(obs_2)
        }

        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        if reward_manager is not None:
            reward_manager.process_breakdown(env, 1 / env.fps)

        if video_path is not None:
            img = env.render()
            # for i, (name, reward) in enumerate(reward_breakdown.items()):
            #     img = add_reward_to_frame(img, i, name, reward)
            writer.writeFrame(img)
            del img

        if terminated or truncated:
            break
        #env.show_image(img)

    if video_path is not None:
        writer.close()

    env.close()


    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)
    player1_result = determine_win(player_1_stats, player_2_stats)


    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=player1_result
    )

    del env

    return match_stats

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainLogging(Enum):
    NONE = 0
    TO_FILE = 1
    PLOT = 2

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    weights = np.repeat(1.0, 50) / 50
    print(weights, y)
    y = np.convolve(y, weights, "valid")
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(f"{log_folder}/reward_plot.png")
    plt.show()

def train(agent: Agent,
          reward_manager: RewardManager,
          save_handler: Optional[SaveHandler]=None,
          opponent_cfg: OpponentsCfg=OpponentsCfg(),
          resolution: CameraResolution=CameraResolution.LOW,
          train_timesteps: int=400_000,
          train_logging: TrainLogging=TrainLogging.PLOT,
          transitions=None,
          il_type: Union[None, str] = None,
          ):
    # Create environment
    env = SelfPlayWarehouseBrawl(reward_manager=reward_manager,
                                 opponent_cfg=opponent_cfg,
                                 save_handler=save_handler,
                                 resolution=resolution
                                 )
    reward_manager.subscribe_signals(env.raw_env)
    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)

    # Get the base WarehouseBrawl environment
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    try:
        agent.get_env_info(base_env)
        base_env.on_training_start()  # Call on_training_start on the base env

        if il_type == 'bc':
            assert transitions is not None, "BC requires expert demonstrations"
            rng = np.random.default_rng(0)
            bc_trainer = BC(
                observation_space=base_env.observation_space,
                action_space=base_env.action_space,
                demonstrations=transitions,
                rng=rng,
                policy=agent.model.policy,  # Use PPO's policy directly
            )
            bc_trainer.train(n_epochs=15)
            print("BC training complete", end='\n\n\n\n')
        elif il_type == 'gail':
            assert transitions is not None, "GAIL requires expert demonstrations"
            rng = np.random.default_rng(0)
            gail_env = make_vec_env(lambda: SelfPlayWarehouseBrawl(reward_manager=reward_manager,
                                 opponent_cfg=opponent_cfg,
                                 save_handler=save_handler,
                                 resolution=resolution
                                 ), n_envs=1)

            reward_net = BasicRewardNet(
                observation_space=gail_env.observation_space,
                action_space=gail_env.action_space,
                normalize_input_layer=RunningNorm,
            )
            # env.num_envs = 1

            gail_trainer = GAIL(
                demonstrations=transitions,
                demo_batch_size=1024,
                gen_replay_buffer_capacity=512,
                n_disc_updates_per_round=8,
                venv=gail_env,
                gen_algo=agent.model,
                reward_net=reward_net,
                allow_variable_horizon=True,
                # rng=rng,
            )

            # evaluate the learner before training
            # learner_rewards_before_training, _ = evaluate_policy(
            #     agent.model, env, 100, return_episode_rewards=True,
            # )

            # train the learner and evaluate again
            gail_trainer.train(500000)                   # max train steps

            # learner_rewards_after_training, _ = evaluate_policy(
            #     agent.model, env, 100, return_episode_rewards=True,
            # )

            # print("mean reward after training:", np.mean(learner_rewards_after_training))
            # print("mean reward before training:", np.mean(learner_rewards_before_training))



        if train_timesteps > 0:
            agent.learn(env, total_timesteps=train_timesteps, verbose=1)
        base_env.on_training_end()  # Call on_training_end on the base env
    except KeyboardInterrupt:
        pass

    env.close()

    if save_handler is not None:
        save_handler.save_agent()

    if train_logging == TrainLogging.PLOT:
        plot_results(log_dir)

# Example usage:
if __name__ == "__main__":
    # current_agent: your current agent (e.g., loaded from a Stable Baselines3 model)
    # opponent_pool: list of older agent versions (these could be snapshots saved during training)

    # EXAMPLE USAGE
#     action_sheet = [
#     (10, ['a']),
#     (1, ['l']),
#     (20, ['a']),
#     (3, ['a', 'j']),
#     (30, []),
#     (7, ['d']),
#     (1, ['a']),
#     (4, ['a','l']),
#     (1, ['a']),
#     (4, ['a','l']),
#     (1, ['a']),
#     (4, ['a','k']),
#     (20, []),
#     (4, ['d','k']),
#     (20, []),
#     (15, ['space']),
#     (5, []),
#     (15, ['space']),
#     (5, []),
#     (15, ['space']),
#     (5, []),
#     (15, ['space']),
#     (5, []),
#     (15, ['space']),
#     (5, []),
# ]

    # Make sure your agents are properly initialized (or if provided as callables, call them once).
    action_sheet = [
    (5, ['d']),
    (20, []),
    # (17, ['d']),
    # (2, ['space']),
    # (2, ['a']),
    # (10, []),
    (42, ['d']),
    (10, []),
    (1, ['a']),
    (10, []),

    # (2, ['j']),
    # (40, []),
    # (2, ['d','k']),
    # (40, []),
    # (15, ['space']),
    # (5, []),
    
    # (10, []),
    # (20, ['space']),
    # (10, []),
    # (15, ['d','space']),
    # (10, []),
    # (20, ['space']),
    # (10, []),
    # (15, ['a','space']),
    # (10, []),
    # (5, ['d','l']),
    # (10, []),
    # (5, ['a','l']),
    # (10, []),

    ]

    action_sheet2 = [
        # (5, ['a']),
        (67, []),
        (2, ['g']),
        (5, []),
    ]
    

    opponent = BasedBetterAgent() 
    current_agent = BasedBetterAgent()  # Replace with your current agent initialization
    reward_manager = RewardManager(reward_functions, signal_subscriptions)

    run_match(opponent, current_agent, video_path="testing.mp4", resolution=CameraResolution.LOW, reward_manager=reward_manager)