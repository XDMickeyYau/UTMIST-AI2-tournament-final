import itertools
import json
import os
from typing import Optional, List
from functools import partial

from stable_baselines3 import PPO
from ComboBasedAgent import ComboBasedAgent
from ComboBasedAgent_random import ComboBasedAgent_Random
from DodgeBasedAgent import DodgeBasedAgent
from GroundPoundBasedAgent import  GroundPoundBasedAgent
from agents import Agent, BasedAgent, BasedBetterAgent, ClockworkAgent, ConstantAgent, ConstantEdgeJumpAgent, RandomAgent, SB3Agent, SB3AgentRuled
from agents_reduced import SB3AgentReduced
from brawl_env import MatchStats, WarehouseBrawl, Result
from camera import CameraResolution
import skvideo
import skvideo.io

from reward_manager import RewardManager
from reward import reward_functions, signal_subscriptions
from run_agent import run_match
import csv

import pandas as pd

def update_elo(R_a: float, R_b: float, R_a_hist:List[float],  R_b_hist:List[float], outcome: float, K: int = 32) -> tuple[float, float, List[float], List[float]]:
    """
    Update Elo ratings for two players.

    R_a: Rating of agent A (current agent).
    R_b: Rating of agent B (the opponent).
    outcome: 1.0 if agent A wins, 0.0 if agent A loses (draws can be 0.5).
    K: The K-factor determining maximum rating change per match.
    Returns:
      (new_R_a, new_R_b): Updated ratings.
    """
    ea = 1 / (1 + 10 ** ((R_b - R_a) / 400))
    eb = 1 / (1 + 10 ** ((R_a - R_b) / 400))
    R_a = R_a + K * (outcome - ea)
    R_b = R_b + K * ((1-outcome) - eb)
    R_a_hist.append(R_a)
    R_b_hist.append(R_b)

    return R_a, R_b, R_a_hist, R_b_hist


def evaluate_against_pool(current_agent, opponent, num_matches: int = 20,
                          initial_elo: float = 1200.0, K: int = 32):
    """
    Evaluate the current_agent against the opponent.
    in a round-robin style. Elo is initialized once here for the current agent,
    and opponent is also initialized with the same starting Elo. 

    Parameters:
        current_agent: The agent to be evaluated (must have a .predict() method).
        opponent: The opponent agent.
        num_matches: Number of matches to play in total.
        initial_elo: The starting Elo rating (commonly 1200).
        K: K-factor for Elo updates.

    Returns:
        overall_win_rate: Overall win rate of current_agent over all matches.
        final_elo: Final Elo rating for current_agent after all matches.
        opponent_elo: Final Elo rating for opponent after all matches.
    """
    total_wins = 0
    total_draws = 0
    total_losses = 0

    # Initialize current agent's Elo rating
    elo_current = initial_elo
    elo_opponent = initial_elo

    for i in range(num_matches):
        # Run a single match between current_agent and this opponent
        match_stats = run_match(current_agent, opponent)
        # Assume match_stats.player1_result is Result.WIN for a win.
        if match_stats.player1_result == Result.WIN:
            outcome = 1.0
            total_wins += 1
        elif match_stats.player1_result == Result.DRAW:
            outcome = 0.5
            total_draws += 1
        else:
            outcome = 0.0
            total_losses += 1

        # Update Elo ratings for the pair
        elo_current, elo_opponent, _, _  = update_elo(elo_current, elo_opponent, [], [], outcome, K)

    overall_win_rate = total_wins / num_matches 
    # opponent_elos[idx] = elo_opponent  # You can also use the opponent's identity instead of index
    print(f"  Matches: {num_matches}, Win rate vs. opponent: {overall_win_rate * 100:.1f}%, "
            f"Current Agent Elo: {elo_current:.1f}, Opponent Elo: {elo_opponent:.1f}")

    
    return total_wins, total_draws, total_losses, elo_current, elo_opponent

# Example usage:
if __name__ == "__main__":

    import argparse

    # ----------------- Parse arguments -----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_steps", type=int, default=0, help="Number of steps to load from pretrained model.")
    parser.add_argument("--version", type=str, default="", help="Version of the model.")
    parser.add_argument("--ent_coef", type=float, default=0., help="Entropy coefficient.")
    parser.add_argument("--il", type=str, default="", help="Imitation learning.", choices=["", "il", "dodge","betterbased"])
    parser.add_argument("--agent_class", type=str, default="SB3Agent", help="Agent class")
    parser.add_argument("--suffix", type=str, default="", help="Base opponent to run against.")
    parser.add_argument("--iter", type=str, default="", help="Base opponent to run against.")
    parser.add_argument("--video", action="store_true", help="Enable video output")
    args = parser.parse_args()

    # ----------------- Print arguments -----------------
    pretrained_steps = args.pretrained_steps
    version = args.version
    ent_coef = args.ent_coef
    il = args.il
    suffix = args.suffix
    agent_class_str = args.agent_class
    iter = args.iter
    video = args.video

    print(f"Pretrained steps: {pretrained_steps}")
    print(f"Version: {version}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"IL: {il}")
    print(f"Suffix: {suffix}")
    print(f"Agent class: {agent_class_str}")
    print(f"Iteration: {iter}")
    print(f"Video: {video}")
    # ----------------- Choose the agent class -----------------
    if agent_class_str == 'SB3Agent':
        agent_class = partial(SB3Agent,sb3_class=PPO)
    elif agent_class_str == 'SB3AgentReduced':
        agent_class = partial(SB3AgentReduced,sb3_class=PPO)
    elif agent_class_str == 'SB3AgentRuled':
        agent_class = partial(SB3AgentRuled,sb3_class=PPO)
    else:
        raise ValueError(f"Unknown agent class: {agent_class_str}")

    print(f"Agent class: {agent_class}")

     # ----------------- Choose the model to run -----------------
    def get_chosen_name(version, pretrained_steps, il, ent_coef, suffix=suffix):
        chosen_name = f'ver{version}ppo_{(pretrained_steps) // 1000}k'
        if il:
            chosen_name += f'_{il}'
        if ent_coef:
            chosen_name += f'_ent_coef_{ent_coef}'
        if agent_class_str == 'SB3AgentReduced':
            chosen_name += '_reduced'
        if suffix:
            chosen_name += f'_{suffix}'
        return chosen_name

    chosen_name = get_chosen_name(version, pretrained_steps, il, ent_coef, suffix)
    print(f"Chosen name: {chosen_name}")

    # ----------------- Choose the checkpoint to run -----------------
    def get_checkpoint_path(pretrained_steps, il, chosen_name):
        checkpoint_paths = {
            0: {
                "dodge": 'checkpoints/experiment_bc_dodge_3/rl_model_0_steps.zip',
                "il" : 'checkpoints/experiment_5/rl_model_2048_steps.zip',
                "": None
            },
            1_000_000: {
                "betterbased": f'checkpoints/{chosen_name}/rl_model_1004400_steps.zip',
                "dodge": f"checkpoints/{chosen_name}/rl_model_1004400_steps.zip",
                "il": f'checkpoints/{chosen_name}/rl_model_1006448_steps.zip',
                "": f'checkpoints/{chosen_name}/rl_model_1001472_steps.zip'
            },
            2_000_000: {
                "betterbased": f'checkpoints/{chosen_name}/rl_model_2008800_steps.zip',
                "il": f'checkpoints/{chosen_name}/rl_model_2010848_steps.zip',
                "": f'checkpoints/{chosen_name}/rl_model_2005872_steps.zip'
            },
            3_000_000: {
                "il": f'checkpoints/{chosen_name}/rl_model_3015248_steps.zip',
                "": f'checkpoints/{chosen_name}/rl_model_3010272_steps.zip'
            },
            4_000_000: {
                "il": f'checkpoints/{chosen_name}/rl_model_4019648_steps.zip',
            },
        }

        return checkpoint_paths[pretrained_steps][il]

    checkpoints_path = get_checkpoint_path(pretrained_steps, il, chosen_name)
    print(f"Checkpoints path: {checkpoints_path}")
    
    # ----------------- Choose the base opponent to run -----------------
    base_candidates = {
        'constant': ConstantAgent(),
        'random': RandomAgent(),
        'based': BasedAgent(),
        'basedbetter': BasedBetterAgent(),
        'dodgebased': DodgeBasedAgent(),
        'combobased_random': ComboBasedAgent_Random(),
        'combobased': ComboBasedAgent(),
        'ConstantEdgeJumpAgent': ConstantEdgeJumpAgent(),
        'groundpound_based': GroundPoundBasedAgent(),
        'submit_1': SB3AgentRuled(sb3_class=PPO, file_path='checkpoints/ver100ppo_2000k_il_ent_coef_0.001_4_mixed/rl_model_2010848_steps.zip'),

        # 'ver8ppo_2000k_il_e0.005': SB3Agent(sb3_class=PPO, file_path='checkpoints/ver8ppo_2000k_il_ent_coef_0.005/rl_model_2010848_steps.zip'),
        # 'ver8ppo_1000k_il_e0.005': SB3Agent(sb3_class=PPO, file_path='checkpoints/ver8ppo_1000k_il_ent_coef_0.005/rl_model_1006448_steps.zip'),
        # 'ver8ppo_2000k_il': SB3Agent(sb3_class=PPO, file_path='checkpoints/ver8ppo_2000k_il/rl_model_2010848_steps.zip'),
        # 'ver7ppo_2000k_il': SB3Agent(sb3_class=PPO, file_path='checkpoints/ver7ppo_2000k_il/rl_model_1006448_steps.zip'),
        # 'ver7ppo_1000k_il_e0.005': SB3Agent(sb3_class=PPO, file_path='checkpoints/ver7ppo_1000k_il_ent_coef_0.005/rl_model_1006448_steps.zip'),
    }
    
    # ----------------- Choose the pretrained opponent to run -----------------
    opponent_versions = [10, 9, 8]
    opponent_il = "il"
    opponent_ent_coef = 0.005
    opponent_suffix = None
    for opponent_version in opponent_versions:
        opponent_name = get_chosen_name(opponent_version, pretrained_steps, opponent_il, opponent_ent_coef, opponent_suffix)
        opponent_path = get_checkpoint_path(pretrained_steps, opponent_il, opponent_name)
        base_candidates[opponent_name+"_ruled"] = SB3AgentRuled(sb3_class=PPO, file_path=opponent_path)
        # base_candidates[opponent_name] = SB3Agent(sb3_class=PPO, file_path=opponent_path)
        print(f"Added {opponent_name} to base candidates with path {opponent_path}")


    # base_candidates[chosen_name] = SB3Agent(sb3_class=PPO, file_path=checkpoints_path)
    
    print(f"Base candidates: {base_candidates}")
    
    if agent_class_str == 'SB3AgentRuled':
        chosen_name += '_ruled'
    candidates = {
        chosen_name: agent_class(file_path=checkpoints_path),
    }

    print(f"Candidates: {candidates}")
    
    
    # ----------------- Run the evaluation -----------------
    reward_manager = RewardManager(reward_functions, signal_subscriptions)
    evaluation_folder = f"evaluations/evaluation_ver{version}_{(pretrained_steps) // 1000}k_{iter}"
    os.makedirs(evaluation_folder, exist_ok=True)

    # Get all possible combinations of base vs candidates
    combinations = list(itertools.product(candidates.items(), base_candidates.items()))

    if not video:
        results = []
        # Get all possible combinations of any 2 entries
        # combinations += list(itertools.combinations(candidates.items(), 2))
        for (name1, current_agent), (name2, opponent) in combinations:
            print(f"Comparing {name1} against {name2}")
            total_wins, total_draws, total_losses, my_elo, opp_elo = evaluate_against_pool(current_agent, opponent, num_matches=100)
            print(f"Overall win : {total_wins}, draw : {total_draws}, loss : {total_losses}, Elo: {my_elo:.1f}, Opponent Elo: {opp_elo:.1f}")
            result = {
                "agent1": name1,
                "agent2": name2,
                "total_wins": total_wins,
                "total_draws": total_draws,
                "total_losses": total_losses,
                "elo": my_elo,
                "opponent_elo": opp_elo,
            }
            json.dump(result, open(f"{evaluation_folder}/{name1}_VS_{name2}.json", "w"))
            results.append(result)

        # check all
        try:
            # List to hold data from all JSON files
            data_list = []

            # Iterate over all files in the evaluation folder
            for filename in os.listdir(evaluation_folder):
                if filename.endswith(".json"):
                    file_path = os.path.join(evaluation_folder, filename)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        data_list.append(data)

            # Combine all data into a single DataFrame
            df = pd.DataFrame(data_list)

            df.sort_values(by=["agent1", "agent2"], ascending=True, inplace=True)

            # Save the DataFrame to a CSV file
            df.to_csv(f"{evaluation_folder}/evaluation.csv", index=False)

            print(f"Combined CSV saved to {evaluation_folder}/evaluation.csv")
        except Exception as e:
            print(f"Error combining JSON files: {e}")


    if video:
        for (name1, current_agent), (name2, opponent) in combinations:
            for i in range(3):
                run_match(current_agent, opponent, agent_1_name=name1.replace("ppo_", ""), agent_2_name=name2.replace("ppo_", ""), video_path=f"{evaluation_folder}/{name1}_VS_{name2}_{i}.mp4", resolution=CameraResolution.LOW, reward_manager=reward_manager)
                print(f"Match video saved as {name1}_VS_{name2}.mp4")




