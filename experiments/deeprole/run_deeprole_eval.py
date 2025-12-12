#!/usr/bin/env python3
"""Quick evaluation script for DeepRole agent."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import SimpleDeepRoleAgent


def main():
    """Run quick evaluation."""

    # Use the actual trained networks
    project_root = Path(__file__).parent.parent.parent
    networks_path = project_root / "agents" / "deeprole" / "trained_networks.pkl"

    if networks_path.exists():
        print(f"Using networks: {networks_path}")
    else:
        print(f"Warning: Networks not found at {networks_path}")
        networks_path = str(networks_path)  # Convert to string anyway

    # Get number of games from command line or default to 10
    import sys
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"Running {num_games} test games...")
    print()

    # Test configuration: Use full DeepRole agents with belief tracking
    from agents.deeprole.deeprole_agent import DeepRoleAgent
    agent_classes = [
        lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2),  # P0 - reduced for testing
        SimpleRandomAgent,  # P1
        lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2),  # P2 - reduced for testing
        SimpleRandomAgent,  # P3
        SimpleRandomAgent,  # P4
    ]

    results = evaluate_agents(
        agent_classes,
        num_games=num_games,
        verbose=False,  # Less verbose output
        seed=42
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    total_games = results['lib_wins'] + results['fasc_wins']
    print(f"Liberal wins: {results['lib_wins']}/{total_games} ({results['lib_wins']/total_games*100:.1f}%)")
    print(f"Fascist wins: {results['fasc_wins']}/{total_games} ({results['fasc_wins']/total_games*100:.1f}%)")

    # Show average rewards
    print("\nAverage rewards:")
    for player, rewards in results["player_rewards"].items():
        role = "DeepRole" if int(player[1]) in [0, 2] else "Random"
        avg = sum(rewards) / len(rewards) if rewards else 0
        print(f"  {player} ({role}): {avg:+.3f}")

    # Report CFR statistics from DeepRole agents
    print("\n" + "=" * 50)
    print("CFR STATISTICS")
    print("=" * 50)

    # Create temporary agents to check their stats
    p0_agent = DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=2)
    if hasattr(p0_agent, 'action_stats'):
        total = p0_agent.action_stats.get('total', 0)
        if total > 0:
            print("Note: Statistics only from last agent instance")
        else:
            print("No CFR statistics available (agents recreated each game)")


if __name__ == "__main__":
    main()