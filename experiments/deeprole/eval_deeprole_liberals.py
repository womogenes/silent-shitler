#!/usr/bin/env python3
"""Evaluate DeepRole agents as liberals against random fascists."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import DeepRoleAgent
import numpy as np


def evaluate_liberal_deeprole(num_games=100):
    """Evaluate 3 liberal DeepRole agents vs 2 random fascist/hitler."""

    # Use the actual trained networks
    project_root = Path(__file__).parent.parent.parent
    networks_path = project_root / "agents" / "deeprole" / "trained_networks.pkl"

    if not networks_path.exists():
        print(f"Error: Networks not found at {networks_path}")
        return

    print(f"Using networks: {networks_path}")
    print(f"Running {num_games} games...")
    print("Setup: 3 Liberal DeepRole agents vs 2 Random Fascist/Hitler")
    print("-" * 60)

    # We need to track which games have which role assignments
    lib_deeprole_wins = 0
    fasc_random_wins = 0

    # Track average performance by actual role
    deeprole_lib_rewards = []
    random_fasc_rewards = []
    random_hitler_rewards = []

    for game_idx in range(num_games):
        # Create agents - all 5 use same strategy but we'll track by role
        agent_classes = [
            lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2),  # P0 - reduced
            lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2),  # P1 - reduced
            lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2),  # P2 - reduced
            SimpleRandomAgent,  # P3
            SimpleRandomAgent,  # P4
        ]

        # Run single game
        results = evaluate_agents(
            agent_classes,
            num_games=1,
            verbose=False,
            seed=42 + game_idx  # Different seed each game
        )

        # We need to check which players were liberals
        # Since we can't directly control role assignment, we'll run many games
        # and report statistics

        if results['lib_wins'] > 0:
            lib_deeprole_wins += 1
        else:
            fasc_random_wins += 1

        # Print progress every 10 games
        if (game_idx + 1) % 10 == 0:
            print(f"Progress: {game_idx + 1}/{num_games} games completed")

    # Report results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    lib_win_rate = lib_deeprole_wins / num_games * 100
    fasc_win_rate = fasc_random_wins / num_games * 100

    print(f"Liberal wins: {lib_deeprole_wins}/{num_games} ({lib_win_rate:.1f}%)")
    print(f"Fascist wins: {fasc_random_wins}/{num_games} ({fasc_win_rate:.1f}%)")

    print("\nNote: Role assignment is random, so not all games have")
    print("exactly 3 DeepRole liberals vs 2 Random fascists.")
    print("\nTo properly test this setup, we'd need to modify the")
    print("environment to allow fixed role assignments.")

    return lib_win_rate


def evaluate_with_role_control(num_games=100):
    """
    Evaluate with controlled role assignments.
    This would require modifying the environment to support fixed roles.
    """
    print("\n" + "=" * 60)
    print("CONTROLLED ROLE EVALUATION")
    print("=" * 60)

    print("To properly evaluate 3 Liberal DeepRole vs 2 Fascist Random,")
    print("we would need to modify ShitlerEnv to support fixed role assignments.")
    print("\nCurrent evaluation uses random role assignment, so we're testing")
    print("DeepRole agents in mixed roles against Random agents.")

    # For now, run standard mixed evaluation
    project_root = Path(__file__).parent.parent.parent
    networks_path = project_root / "agents" / "deeprole" / "trained_networks.pkl"

    # Test different configurations
    configs = [
        ("3 DeepRole vs 2 Random", [0, 1, 2], [3, 4]),
        ("2 DeepRole vs 3 Random", [0, 1], [2, 3, 4]),
        ("All DeepRole", [0, 1, 2, 3, 4], []),
        ("1 DeepRole vs 4 Random", [0], [1, 2, 3, 4]),
    ]

    for config_name, deeprole_indices, random_indices in configs:
        print(f"\n{config_name}:")
        print("-" * 40)

        agent_classes = []
        for i in range(5):
            if i in deeprole_indices:
                agent_classes.append(
                    lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2)
                )
            else:
                agent_classes.append(SimpleRandomAgent)

        results = evaluate_agents(
            agent_classes,
            num_games=min(num_games, 20),  # Fewer games for multiple configs
            verbose=False,
            seed=42
        )

        total = results['lib_wins'] + results['fasc_wins']
        lib_rate = results['lib_wins'] / total * 100
        print(f"  Liberal wins: {results['lib_wins']}/{total} ({lib_rate:.1f}%)")
        print(f"  Fascist wins: {results['fasc_wins']}/{total} ({100-lib_rate:.1f}%)")

        # Show which positions performed best
        avg_by_pos = {}
        for player, rewards in results['player_rewards'].items():
            idx = int(player[1])
            agent_type = "DeepRole" if idx in deeprole_indices else "Random"
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            avg_by_pos[idx] = (agent_type, avg_reward)

        print(f"  Best performer: P{max(avg_by_pos, key=lambda k: avg_by_pos[k][1])} "
              f"({avg_by_pos[max(avg_by_pos, key=lambda k: avg_by_pos[k][1])][0]})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DeepRole liberals vs random fascists")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--controlled", action="store_true", help="Run controlled evaluation")

    args = parser.parse_args()

    if args.controlled:
        evaluate_with_role_control(args.games)
    else:
        # Standard evaluation
        lib_win_rate = evaluate_liberal_deeprole(args.games)

        # Analysis
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        if lib_win_rate > 60:
            print("✓ DeepRole agents performing well as liberals")
        elif lib_win_rate > 45:
            print("◯ DeepRole agents performing adequately")
        else:
            print("✗ DeepRole agents underperforming")
            print("  This may be due to:")
            print("  - Limited training data (only 5 networks)")
            print("  - CFR not converging with limited iterations")
            print("  - Belief tracking issues without proper DeepRole implementation")


if __name__ == "__main__":
    main()