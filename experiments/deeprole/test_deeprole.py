#!/usr/bin/env python3
"""Test DeepRole agent against random agents."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import DeepRoleAgent, SimpleDeepRoleAgent


def test_deeprole_vs_random(num_games=100, use_simple=False):
    """Test DeepRole agent against random agents.

    Args:
        num_games: Number of games to simulate
        use_simple: Whether to use SimpleDeepRoleAgent for faster play
    """
    print("=" * 60)
    print("DEEPROLE AGENT EVALUATION")
    print("=" * 60)

    # Check if trained networks exist
    networks_path = project_root / "trained_networks.pkl"
    if not networks_path.exists():
        networks_path = project_root / "checkpoints" / "networks_partial.pkl"
        if not networks_path.exists():
            networks_path = project_root / "trained_networks_test.pkl"
            if not networks_path.exists():
                networks_path = project_root / "agents" / "deeprole" / "trained_networks.pkl"

    if networks_path.exists():
        print(f"Using networks from: {networks_path}")
    else:
        print("Warning: No trained networks found, will use random play")
        networks_path = "trained_networks.pkl"  # Will fail gracefully

    # Create agent classes
    if use_simple:
        print("Using SimpleDeepRoleAgent (faster)")
        DeepRoleClass = lambda: SimpleDeepRoleAgent(networks_path)
    else:
        print("Using full DeepRoleAgent (slower but more accurate)")
        print("  CFR iterations: 50")
        print("  Max depth: 3")
        DeepRoleClass = lambda: DeepRoleAgent(networks_path, cfr_iterations=50, max_depth=3)

    # Test different configurations
    configs = [
        ("DeepRole as Liberal (P0) vs Random",
         [DeepRoleClass, SimpleRandomAgent, SimpleRandomAgent, SimpleRandomAgent, SimpleRandomAgent]),

        ("DeepRole as Fascist (P1) vs Random",
         [SimpleRandomAgent, DeepRoleClass, SimpleRandomAgent, SimpleRandomAgent, SimpleRandomAgent]),

        ("All DeepRole agents",
         [DeepRoleClass, DeepRoleClass, DeepRoleClass, DeepRoleClass, DeepRoleClass]),

        ("DeepRole team (P0,P2) vs Random",
         [DeepRoleClass, SimpleRandomAgent, DeepRoleClass, SimpleRandomAgent, SimpleRandomAgent]),
    ]

    for config_name, agent_classes in configs:
        print(f"\n{config_name}")
        print("-" * 40)

        results = evaluate_agents(
            agent_classes,
            num_games=num_games,
            verbose=False,
            seed=42
        )

        # Print results
        total_games = results["lib_wins"] + results["fasc_wins"]
        lib_rate = results["lib_wins"] / total_games * 100
        fasc_rate = results["fasc_wins"] / total_games * 100

        print(f"  Liberal wins: {results['lib_wins']}/{total_games} ({lib_rate:.1f}%)")
        print(f"  Fascist wins: {results['fasc_wins']}/{total_games} ({fasc_rate:.1f}%)")

        # Calculate average rewards for each player
        print(f"  Average rewards by player:")
        for player, rewards in results["player_rewards"].items():
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                print(f"    {player}: {avg_reward:+.3f}")


def test_single_game(verbose=True):
    """Run a single game with verbose output for debugging."""
    print("=" * 60)
    print("SINGLE GAME TEST (VERBOSE)")
    print("=" * 60)

    project_root = Path(__file__).parent.parent.parent
    networks_path = project_root / "trained_networks.pkl"
    if not networks_path.exists():
        networks_path = project_root / "checkpoints" / "networks_partial.pkl"

    # Create mixed agents
    agent_classes = [
        lambda: DeepRoleAgent(networks_path, cfr_iterations=10, max_depth=3),  # P0 - DeepRole
        SimpleRandomAgent,  # P1 - Random
        SimpleRandomAgent,  # P2 - Random
        lambda: SimpleDeepRoleAgent(networks_path),  # P3 - Simple DeepRole
        SimpleRandomAgent,  # P4 - Random
    ]

    # Run single game
    from shitler_env.game import ShitlerEnv

    env = ShitlerEnv()
    env.reset(seed=42)

    # Show initial roles
    print(f"\nInitial setup:")
    print(f"  Roles: {env.roles}")
    print(f"  Liberals: {[p for p, r in env.roles.items() if r == 'lib']}")
    print(f"  Fascists: {[p for p, r in env.roles.items() if r == 'fasc']}")
    print(f"  Hitler: {[p for p, r in env.roles.items() if r == 'hitty']}")
    print()

    # Create agents
    agents = {f"P{i}": agent_classes[i]() for i in range(5)}

    # Play game
    step = 0
    while not all(env.terminations.values()):
        agent_name = env.agent_selection
        agent = agents[agent_name]
        obs = env.observe(agent_name)

        if verbose and step < 50:  # Limit verbosity
            print(f"Step {step}: {agent_name} in phase '{obs['phase']}'")

        action = agent.get_action(obs)
        env.step(action)
        step += 1

    # Show results
    print(f"\nGame ended after {step} steps")
    print(f"Final scores: {env.rewards}")

    winner = "liberals" if env.rewards["P0"] > 0 and env.roles["P0"] == "lib" else "fascists"
    print(f"Winner: {winner}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DeepRole agent")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to simulate")
    parser.add_argument("--simple", action="store_true",
                        help="Use SimpleDeepRoleAgent for faster play")
    parser.add_argument("--single", action="store_true",
                        help="Run single verbose game for debugging")

    args = parser.parse_args()

    if args.single:
        test_single_game()
    else:
        test_deeprole_vs_random(args.games, args.simple)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("\nNote: DeepRole performance depends on:")
    print("  1. Quality of trained networks")
    print("  2. CFR iterations (more = better but slower)")
    print("  3. Search depth (deeper = better but slower)")
    print("  4. Belief tracking accuracy")