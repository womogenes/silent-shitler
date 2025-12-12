#!/usr/bin/env python
"""Clean evaluation using full state serialization for DeepRole."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
from shitler_env.game import ShitlerEnv
from shitler_env.agent import SimpleRandomAgent as RandomAgent
from agents.deeprole.deeprole_agent_v2 import DeepRoleAgentV2


def evaluate_with_state_access(lib_agents, fasc_agents, num_games=100, seed=None, verbose=False):
    """Evaluate agents with full state access support.

    This is a clean evaluation function that provides full game state
    to agents that need it (like DeepRole with CFR).

    Args:
        lib_agents: List of liberal agent instances
        fasc_agents: List of fascist/hitler agent instances
        num_games: Number of games to simulate
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Results dictionary
    """
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "games_played": 0
    }

    for game_num in tqdm(range(num_games), desc="Games", disable=not verbose):
        # Create environment
        env = ShitlerEnv()
        game_seed = None if seed is None else seed + game_num
        env.reset(seed=game_seed)

        # Assign agents to players
        # For simplicity, first 3 are liberals, last 2 are fascists
        # (In a real game, roles are random, but for testing this is fine)
        game_agents = {}
        for i in range(3):
            agent = lib_agents[i % len(lib_agents)]
            if hasattr(agent, 'reset'):
                agent.reset(player_idx=i)
            game_agents[f"P{i}"] = agent

        for i in range(3, 5):
            agent = fasc_agents[(i-3) % len(fasc_agents)]
            if hasattr(agent, 'reset'):
                agent.reset(player_idx=i)
            game_agents[f"P{i}"] = agent

        # Play game
        while not all(env.terminations.values()):
            current_agent_name = env.agent_selection
            current_agent = game_agents[current_agent_name]

            # Get observation
            obs = env.observe(current_agent_name)

            # Get action - provide full state if agent can use it
            if isinstance(current_agent, DeepRoleAgentV2):
                # Provide full state for clean CFR simulation
                game_state = env.get_state_dict()
                action = current_agent.get_action(
                    obs,
                    game_state=game_state,
                    agent_name=current_agent_name
                )
            else:
                # Regular agent - just observation
                action = current_agent.get_action(
                    obs,
                    agent_name=current_agent_name
                )

            # Take action
            env.step(action)

        # Record results
        # Check who won based on rewards
        lib_win = any(env.rewards[f"P{i}"] > 0 for i in range(3))
        if lib_win:
            results["lib_wins"] += 1
        else:
            results["fasc_wins"] += 1
        results["games_played"] += 1

    return results


def main():
    """Run clean evaluation of DeepRole v2."""
    print("=" * 60)
    print("DeepRole V2 Evaluation - Clean State Serialization")
    print("=" * 60)

    # Path to trained networks
    networks_path = Path(__file__).parent.parent.parent / "agents" / "deeprole" / "trained_networks_1000_16_15.pkl"

    if not networks_path.exists():
        print(f"Error: Networks not found at {networks_path}")
        return

    print(f"Networks: {networks_path}")
    print("Creating agents...")

    # Create liberal agents (DeepRole v2)
    lib_agents = []
    for i in range(3):
        agent = DeepRoleAgentV2(
            networks_path=str(networks_path),
            cfr_iterations=10,  # Reduced for speed
            max_depth=2
        )
        lib_agents.append(agent)

    # Create fascist agents (Random)
    fasc_agents = [RandomAgent() for _ in range(2)]

    # Run evaluation
    num_games = 20
    print(f"\nRunning {num_games} games...")
    print("Liberals: DeepRole V2 (with full state access)")
    print("Fascists: Random")

    results = evaluate_with_state_access(
        lib_agents,
        fasc_agents,
        num_games=num_games,
        seed=42,
        verbose=True
    )

    # Print results
    lib_rate = results['lib_wins'] / results['games_played']
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Games played: {results['games_played']}")
    print(f"Liberal wins: {results['lib_wins']} ({lib_rate:.1%})")
    print(f"Fascist wins: {results['fasc_wins']} ({(1-lib_rate):.1%})")

    # Check DeepRole statistics
    if hasattr(lib_agents[0], 'action_stats'):
        stats = lib_agents[0].action_stats
        total = stats['total']
        if total > 0:
            print(f"\nDeepRole Action Stats:")
            print(f"  CFR strategies used: {stats['cfr_strategy']} ({100*stats['cfr_strategy']/total:.1f}%)")
            print(f"  Random fallback: {stats['cfr_fallback']} ({100*stats['cfr_fallback']/total:.1f}%)")
            print(f"  Errors: {stats['cfr_error']} ({100*stats['cfr_error']/total:.1f}%)")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if lib_rate > 0.5:
        print("✓ DeepRole V2 is working well with clean state serialization!")
    elif lib_rate > 0.3:
        print("⚠ DeepRole V2 shows moderate performance.")
    else:
        print("✗ DeepRole V2 may need debugging.")

    print("\nKey improvements in V2:")
    print("- Clean state serialization (no fragile reconstruction)")
    print("- Direct state access for CFR (no reverse engineering)")
    print("- Simpler, more maintainable code")


if __name__ == "__main__":
    main()