"""Evaluate Selfish Agent vs Random Agent baseline.

Compares the selfish algorithm from the ISMCTS Secret Hitler paper
against random play.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shitler_env"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents"))

from game import ShitlerEnv
from agent import SimpleRandomAgent, BaseAgent
from eval_agent import evaluate_agents
import random


class SelfishAgent(BaseAgent):
    """
    Selfish agent that prioritizes its own party's policies.

    Algorithm 1 from "Competing in a Complex Hidden Role Game with
    Information Set Monte Carlo Tree Search" (arxiv:2005.07156)

    For policy selection phases:
    - Liberals try to enact liberal policies / discard fascist
    - Fascists try to enact fascist policies / discard liberal

    For other phases (nomination, voting, execution):
    - Acts randomly (no strategic behavior)
    """

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action using the selfish algorithm."""
        # Determine party affiliation from role
        # role: 0=lib, 1=fasc, 2=hitty
        role = obs.get("role", 0)
        is_liberal = (role == 0)

        # Get valid actions
        valid_actions = self.get_valid_actions(obs)

        # Card selection phases - apply selfish algorithm
        if "card_action_mask" in obs:
            return self._selfish_card_action(obs, is_liberal)

        # Other phases - random from valid actions
        if valid_actions:
            return random.choice(valid_actions)

        # Fallback
        if action_space:
            return action_space.sample()

        return 0

    def _selfish_card_action(self, obs, is_liberal):
        """
        Apply selfish algorithm for card selection.

        Actions:
        - 0 = discard liberal (play fascist if only 2 cards)
        - 1 = discard fascist (play liberal if only 2 cards)
        """
        cards = obs.get("cards", [])
        mask = obs.get("card_action_mask", [1, 1])

        # Decode number of liberal cards
        num_libs = 0
        for i, v in enumerate(cards):
            if v == 1:
                num_libs = i
                break

        # Determine number of fascist cards
        total_cards = 3 if len(cards) == 4 else 2
        num_fascs = total_cards - num_libs

        # Valid actions based on mask
        valid = [i for i, v in enumerate(mask) if v == 1]

        if is_liberal:
            # Liberal wants to discard fascist (action 1)
            if 1 in valid and num_fascs > 0:
                return 1
            if 0 in valid and num_libs > 0:
                return 0
        else:
            # Fascist wants to discard liberal (action 0)
            if 0 in valid and num_libs > 0:
                return 0
            if 1 in valid and num_fascs > 0:
                return 1

        return random.choice(valid) if valid else 0


def run_experiment(num_games=1000, seed=42):
    """Run selfish vs random experiments."""
    results = {}

    print("=" * 60)
    print("SELFISH AGENT EVALUATION")
    print("=" * 60)
    print(f"Games per experiment: {num_games}")
    print(f"Seed: {seed}")
    print()

    # Experiment 1: All Selfish Agents
    print("1. All Selfish Agents")
    print("-" * 40)
    selfish_results = evaluate_agents(
        [SelfishAgent] * 5,
        num_games=num_games,
        verbose=True,
        seed=seed
    )
    results["all_selfish"] = {
        "lib_win_rate": selfish_results["lib_win_rate"],
        "fasc_win_rate": selfish_results["fasc_win_rate"],
    }
    print(f"  Liberal win rate: {selfish_results['lib_win_rate']:.2%}")
    print(f"  Fascist win rate: {selfish_results['fasc_win_rate']:.2%}")
    print()

    # Experiment 2: All Random Agents (baseline)
    print("2. All Random Agents (baseline)")
    print("-" * 40)
    random_results = evaluate_agents(
        [SimpleRandomAgent] * 5,
        num_games=num_games,
        verbose=True,
        seed=seed
    )
    results["all_random"] = {
        "lib_win_rate": random_results["lib_win_rate"],
        "fasc_win_rate": random_results["fasc_win_rate"],
    }
    print(f"  Liberal win rate: {random_results['lib_win_rate']:.2%}")
    print(f"  Fascist win rate: {random_results['fasc_win_rate']:.2%}")
    print()

    # Experiment 3: Selfish Liberals vs Random Fascists
    print("3. Selfish Liberals vs Random Fascists")
    print("-" * 40)
    # We need to assign agents based on role, which we don't know ahead of time
    # So we use a mixed approach: all selfish, but fascists act selfishly too
    # Actually, let's create a custom evaluation for this

    mixed_results_1 = evaluate_role_based_agents(
        liberal_agent_class=SelfishAgent,
        fascist_agent_class=SimpleRandomAgent,
        num_games=num_games,
        seed=seed
    )
    results["selfish_lib_vs_random_fasc"] = {
        "lib_win_rate": mixed_results_1["lib_win_rate"],
        "fasc_win_rate": mixed_results_1["fasc_win_rate"],
    }
    print(f"  Liberal win rate: {mixed_results_1['lib_win_rate']:.2%}")
    print(f"  Fascist win rate: {mixed_results_1['fasc_win_rate']:.2%}")
    print()

    # Experiment 4: Random Liberals vs Selfish Fascists
    print("4. Random Liberals vs Selfish Fascists")
    print("-" * 40)
    mixed_results_2 = evaluate_role_based_agents(
        liberal_agent_class=SimpleRandomAgent,
        fascist_agent_class=SelfishAgent,
        num_games=num_games,
        seed=seed
    )
    results["random_lib_vs_selfish_fasc"] = {
        "lib_win_rate": mixed_results_2["lib_win_rate"],
        "fasc_win_rate": mixed_results_2["fasc_win_rate"],
    }
    print(f"  Liberal win rate: {mixed_results_2['lib_win_rate']:.2%}")
    print(f"  Fascist win rate: {mixed_results_2['fasc_win_rate']:.2%}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<40} {'Lib Win %':>10} {'Fasc Win %':>10}")
    print("-" * 60)
    print(f"{'All Random (baseline)':<40} {results['all_random']['lib_win_rate']*100:>9.1f}% {results['all_random']['fasc_win_rate']*100:>9.1f}%")
    print(f"{'All Selfish':<40} {results['all_selfish']['lib_win_rate']*100:>9.1f}% {results['all_selfish']['fasc_win_rate']*100:>9.1f}%")
    print(f"{'Selfish Libs vs Random Fascs':<40} {results['selfish_lib_vs_random_fasc']['lib_win_rate']*100:>9.1f}% {results['selfish_lib_vs_random_fasc']['fasc_win_rate']*100:>9.1f}%")
    print(f"{'Random Libs vs Selfish Fascs':<40} {results['random_lib_vs_selfish_fasc']['lib_win_rate']*100:>9.1f}% {results['random_lib_vs_selfish_fasc']['fasc_win_rate']*100:>9.1f}%")
    print("=" * 60)

    return results


def evaluate_role_based_agents(liberal_agent_class, fascist_agent_class,
                                num_games=100, seed=None):
    """
    Evaluate with different agent types for different roles.

    Liberals use liberal_agent_class, Fascists/Hitler use fascist_agent_class.
    """
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
    }

    for game_num in range(num_games):
        env = ShitlerEnv()
        game_seed = None if seed is None else seed + game_num
        env.reset(seed=game_seed)

        # Create agents based on actual roles
        agents = {}
        for i, agent_name in enumerate(env.agents):
            role = env.roles[agent_name]
            if role == "lib":
                agents[agent_name] = liberal_agent_class()
            else:  # fasc or hitty
                agents[agent_name] = fascist_agent_class()

        # Play game
        while not all(env.terminations.values()):
            agent_name = env.agent_selection
            obs = env.observe(agent_name)
            action_space = env.action_space(agent_name)
            action = agents[agent_name].get_action(obs, action_space)
            env.step(action)

        # Determine winner
        for agent_name, reward in env.rewards.items():
            if reward == 1:
                role = env.roles[agent_name]
                if role == "lib":
                    results["lib_wins"] += 1
                else:
                    results["fasc_wins"] += 1
                break

        if (game_num + 1) % 100 == 0:
            print(f"  Completed {game_num + 1}/{num_games} games")

    results["lib_win_rate"] = results["lib_wins"] / num_games
    results["fasc_win_rate"] = results["fasc_wins"] / num_games
    results["num_games"] = num_games

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Selfish Agent")
    parser.add_argument("--games", type=int, default=1000, help="Games per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    results = run_experiment(num_games=args.games, seed=args.seed)

    if args.save:
        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"selfish_eval_{timestamp}.json"

        results["metadata"] = {
            "num_games": args.games,
            "seed": args.seed,
            "timestamp": timestamp,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_file}")
