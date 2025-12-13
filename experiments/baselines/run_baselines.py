"""Baseline evaluations: Random vs Random, Selfish vs Selfish."""

import sys
from pathlib import Path
import random
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from agents.random_agent import RandomAgent
from agents.selfish_agent import SelfishAgent

NUM_GAMES = 1000
SEED = 42


def run_eval(agent_class, num_games, seed):
    """Run evaluation with all agents of same type."""
    random.seed(seed)

    wins = {"lib": 0, "fasc": 0}
    win_conds = {
        "lib_5_policies": 0,
        "hitler_executed": 0,
        "fasc_6_policies": 0,
        "hitler_chancellor": 0,
    }

    for i in range(num_games):
        env = ShitlerEnv()
        env.reset(seed=seed + i)
        agent = agent_class()

        while not all(env.terminations.values()):
            obs = env.observe(env.agent_selection)
            action = agent.get_action(obs, env.action_space(env.agent_selection), phase=env.phase)
            env.step(action)

        for name, reward in env.rewards.items():
            if reward == 1:
                role = env.roles[name]
                if role == "lib":
                    wins["lib"] += 1
                    if env.lib_policies >= 5:
                        win_conds["lib_5_policies"] += 1
                    else:
                        win_conds["hitler_executed"] += 1
                else:
                    wins["fasc"] += 1
                    if env.fasc_policies >= 6:
                        win_conds["fasc_6_policies"] += 1
                    else:
                        win_conds["hitler_chancellor"] += 1
                break

    return {
        "liberal_wins": wins["lib"],
        "fascist_wins": wins["fasc"],
        "liberal_win_rate": wins["lib"] / num_games,
        "fascist_win_rate": wins["fasc"] / num_games,
        "win_conditions": win_conds,
        "num_games": num_games,
        "seed": seed,
    }


def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("BASELINE EVALUATIONS")
    print("=" * 50)

    # Random vs Random
    print("\nRunning Random vs Random...")
    random_results = run_eval(RandomAgent, NUM_GAMES, SEED)
    print(f"  Liberal win rate: {random_results['liberal_win_rate']:.1%}")
    print(f"  Win conditions: {random_results['win_conditions']}")

    # Selfish vs Selfish
    print("\nRunning Selfish vs Selfish...")
    selfish_results = run_eval(SelfishAgent, NUM_GAMES, SEED)
    print(f"  Liberal win rate: {selfish_results['liberal_win_rate']:.1%}")
    print(f"  Win conditions: {selfish_results['win_conditions']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "random_vs_random": random_results,
        "selfish_vs_selfish": selfish_results,
        "timestamp": timestamp,
    }

    output_file = results_dir / f"baselines_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
