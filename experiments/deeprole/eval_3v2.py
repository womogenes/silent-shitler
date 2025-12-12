#!/usr/bin/env python3
"""Simple evaluation: 3 DeepRole agents vs 2 Random agents."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import DeepRoleAgent


def main():
    # Use the actual trained networks
    project_root = Path(__file__).parent.parent.parent
    networks_path = project_root / "agents" / "deeprole" / "trained_networks.pkl"

    if not networks_path.exists():
        print(f"Error: Networks not found at {networks_path}")
        return

    print(f"Using networks: {networks_path}")
    print("Configuration: 3 DeepRole (P0,P1,P2) vs 2 Random (P3,P4)")
    print("Note: Roles are randomly assigned, so DeepRole agents may be liberals or fascists")
    print("-" * 60)

    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    # Get CFR parameters from command line
    cfr_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"CFR iterations: {cfr_iter}, Max depth: {max_depth}")

    # Create agents - 3 DeepRole vs 2 Random
    agent_classes = [
        lambda: DeepRoleAgent(networks_path, cfr_iterations=cfr_iter, max_depth=max_depth),  # P0
        lambda: DeepRoleAgent(networks_path, cfr_iterations=cfr_iter, max_depth=max_depth),  # P1
        lambda: DeepRoleAgent(networks_path, cfr_iterations=cfr_iter, max_depth=max_depth),  # P2
        SimpleRandomAgent,  # P3
        SimpleRandomAgent,  # P4
    ]

    print(f"Running {num_games} games...")
    print()

    # Run evaluation
    results = evaluate_agents(
        agent_classes,
        num_games=num_games,
        verbose=False,
        seed=42
    )

    # Report results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    total = results['lib_wins'] + results['fasc_wins']
    lib_rate = results['lib_wins'] / total * 100
    fasc_rate = results['fasc_wins'] / total * 100

    print(f"Liberal wins: {results['lib_wins']}/{total} ({lib_rate:.1f}%)")
    print(f"Fascist wins: {results['fasc_wins']}/{total} ({fasc_rate:.1f}%)")

    # Show average rewards by position
    print("\nAverage rewards by position:")
    for player, rewards in results['player_rewards'].items():
        idx = int(player[1])
        agent_type = "DeepRole" if idx < 3 else "Random"
        avg = sum(rewards) / len(rewards) if rewards else 0
        print(f"  {player} ({agent_type}): {avg:+.3f}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Calculate average for DeepRole vs Random
    deeprole_avg = []
    random_avg = []

    for player, rewards in results['player_rewards'].items():
        idx = int(player[1])
        avg = sum(rewards) / len(rewards) if rewards else 0
        if idx < 3:
            deeprole_avg.append(avg)
        else:
            random_avg.append(avg)

    deeprole_mean = sum(deeprole_avg) / len(deeprole_avg) if deeprole_avg else 0
    random_mean = sum(random_avg) / len(random_avg) if random_avg else 0

    print(f"DeepRole agents average: {deeprole_mean:+.3f}")
    print(f"Random agents average: {random_mean:+.3f}")

    if deeprole_mean > random_mean:
        print("\n✓ DeepRole agents outperforming random agents")
    else:
        print("\n✗ DeepRole agents underperforming")
        print("  Possible reasons:")
        print("  - CFR not converging with limited iterations (10)")
        print("  - Belief tracking disabled (no update mechanism)")
        print("  - Limited training data (only 5 networks)")


if __name__ == "__main__":
    main()