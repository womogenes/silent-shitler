#!/usr/bin/env python
"""Evaluate DeepRole V2 against MetaAgent."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm
import numpy as np
from shitler_env.game import ShitlerEnv
from agents.deeprole.deeprole_agent_v2 import DeepRoleAgentV2
from agents.meta_agent import MetaAgent


def evaluate_matchup(agent_assignments, num_games=100, seed=None, verbose=False):
    """Evaluate a specific matchup configuration.

    Args:
        agent_assignments: Dict mapping player indices to agent instances
        num_games: Number of games to simulate
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Results dictionary with win statistics
    """
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "lib_win_reasons": {},
        "fasc_win_reasons": {},
    }

    for game_num in tqdm(range(num_games), desc="Games", disable=not verbose, ncols=80):
        # Create environment
        env = ShitlerEnv()
        game_seed = None if seed is None else seed + game_num
        env.reset(seed=game_seed)

        # Reset all agents
        for idx, agent in agent_assignments.items():
            if hasattr(agent, 'reset'):
                agent.reset(player_idx=idx)

        # Play game
        while not all(env.terminations.values()):
            current_agent_name = env.agent_selection
            current_idx = int(current_agent_name[1])
            current_agent = agent_assignments[current_idx]

            # Get observation
            obs = env.observe(current_agent_name)

            # Get action - provide full state for DeepRole V2
            if isinstance(current_agent, DeepRoleAgentV2):
                game_state = env.get_state_dict()
                action = current_agent.get_action(
                    obs,
                    game_state=game_state,
                    agent_name=current_agent_name
                )
            else:
                action = current_agent.get_action(
                    obs,
                    agent_name=current_agent_name
                )

            # Take action
            env.step(action)

        # Record results based on actual role assignments
        lib_players = [i for i, role in enumerate(env.roles.values()) if role == "lib"]
        lib_win = any(env.rewards[f"P{i}"] > 0 for i in lib_players)

        if lib_win:
            results["lib_wins"] += 1
            # Determine win reason
            if env.lib_policies >= 5:
                reason = "5 policies"
            else:
                reason = "Hitler executed"
            results["lib_win_reasons"][reason] = results["lib_win_reasons"].get(reason, 0) + 1
        else:
            results["fasc_wins"] += 1
            # Determine win reason
            if env.fasc_policies >= 6:
                reason = "6 policies"
            elif env.phase == "nomination":  # Check for Hitler chancellor
                reason = "Hitler chancellor"
            else:
                reason = "other"
            results["fasc_win_reasons"][reason] = results["fasc_win_reasons"].get(reason, 0) + 1

    return results


def main():
    """Run comprehensive evaluation of DeepRole V2 vs MetaAgent."""
    print("=" * 70)
    print("DEEPROLE V2 vs META AGENT EVALUATION")
    print("=" * 70)

    # Load DeepRole networks
    networks_path = Path(__file__).parent.parent / "agents" / "deeprole" / "trained_networks_1000_16_15.pkl"
    if not networks_path.exists():
        print(f"Error: Networks not found at {networks_path}")
        return

    print(f"Networks: {networks_path}")
    print("Creating agent pool...")

    # Create agent pools
    deeprole_agents = []
    meta_agents = []

    for i in range(3):  # Create 3 of each type
        # DeepRole with moderate CFR iterations
        dr_agent = DeepRoleAgentV2(
            networks_path=str(networks_path),
            cfr_iterations=15,  # Balanced speed/performance
            max_depth=2
        )
        deeprole_agents.append(dr_agent)

        # MetaAgent with default temperature
        meta_agent = MetaAgent(temperature=1.0)
        meta_agents.append(meta_agent)

    print(f"Created {len(deeprole_agents)} DeepRole agents")
    print(f"Created {len(meta_agents)} MetaAgent instances")

    # Test configurations
    configs = [
        {
            "name": "3 DeepRole vs 2 MetaAgent",
            "description": "DeepRole as liberals, MetaAgent as fascists",
            "assignment": lambda: {
                0: deeprole_agents[0],
                1: deeprole_agents[1],
                2: deeprole_agents[2],
                3: meta_agents[0],
                4: meta_agents[1]
            }
        },
        {
            "name": "3 MetaAgent vs 2 DeepRole",
            "description": "MetaAgent as liberals, DeepRole as fascists",
            "assignment": lambda: {
                0: meta_agents[0],
                1: meta_agents[1],
                2: meta_agents[2],
                3: deeprole_agents[0],
                4: deeprole_agents[1]
            }
        },
        {
            "name": "Mixed Teams",
            "description": "Both teams have mixed agents",
            "assignment": lambda: {
                0: deeprole_agents[0],  # Lib
                1: meta_agents[0],      # Lib
                2: deeprole_agents[1],  # Lib
                3: meta_agents[1],      # Fasc
                4: deeprole_agents[2]   # Fasc/Hitler
            }
        }
    ]

    # Run evaluations
    num_games = 50
    all_results = {}

    for config in configs:
        print(f"\n" + "=" * 70)
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 70)
        print(f"Running {num_games} games...")

        # Get agent assignment for this config
        agent_assignment = config["assignment"]()

        # Run evaluation
        results = evaluate_matchup(
            agent_assignment,
            num_games=num_games,
            seed=42,
            verbose=True
        )

        all_results[config["name"]] = results

        # Print results
        total_games = results["lib_wins"] + results["fasc_wins"]
        lib_rate = results["lib_wins"] / total_games if total_games > 0 else 0

        print(f"\nResults:")
        print(f"  Liberal wins: {results['lib_wins']} ({lib_rate:.1%})")
        print(f"  Fascist wins: {results['fasc_wins']} ({(1-lib_rate):.1%})")

        if results["lib_win_reasons"]:
            print(f"  Liberal win reasons:")
            for reason, count in results["lib_win_reasons"].items():
                print(f"    - {reason}: {count}")

        if results["fasc_win_reasons"]:
            print(f"  Fascist win reasons:")
            for reason, count in results["fasc_win_reasons"].items():
                print(f"    - {reason}: {count}")

    # Summary comparison
    print(f"\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    # Compare head-to-head
    dr_as_lib = all_results["3 DeepRole vs 2 MetaAgent"]["lib_wins"]
    meta_as_lib = all_results["3 MetaAgent vs 2 DeepRole"]["lib_wins"]

    print(f"\nAs Liberals (out of {num_games} games):")
    print(f"  DeepRole: {dr_as_lib} wins ({dr_as_lib/num_games:.1%})")
    print(f"  MetaAgent: {meta_as_lib} wins ({meta_as_lib/num_games:.1%})")

    dr_as_fasc = all_results["3 MetaAgent vs 2 DeepRole"]["fasc_wins"]
    meta_as_fasc = all_results["3 DeepRole vs 2 MetaAgent"]["fasc_wins"]

    print(f"\nAs Fascists (out of {num_games} games):")
    print(f"  DeepRole: {dr_as_fasc} wins ({dr_as_fasc/num_games:.1%})")
    print(f"  MetaAgent: {meta_as_fasc} wins ({meta_as_fasc/num_games:.1%})")

    # Overall assessment
    print(f"\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    dr_total = dr_as_lib + dr_as_fasc
    meta_total = meta_as_lib + meta_as_fasc

    print(f"\nTotal wins across both roles:")
    print(f"  DeepRole: {dr_total}/{num_games*2} ({dr_total/(num_games*2):.1%})")
    print(f"  MetaAgent: {meta_total}/{num_games*2} ({meta_total/(num_games*2):.1%})")

    if dr_total > meta_total:
        print("\n✓ DeepRole V2 outperforms MetaAgent overall!")
        print("  The learned CFR strategies are more effective than hand-crafted heuristics.")
    elif meta_total > dr_total:
        print("\n✓ MetaAgent outperforms DeepRole V2!")
        print("  The suspicion-tracking heuristics are very effective.")
    else:
        print("\n✓ DeepRole V2 and MetaAgent are evenly matched!")
        print("  Both approaches have their strengths.")

    # Check mixed team results
    mixed_results = all_results["Mixed Teams"]
    mixed_lib_rate = mixed_results["lib_wins"] / (mixed_results["lib_wins"] + mixed_results["fasc_wins"])
    print(f"\nMixed teams (both agents on each side):")
    print(f"  Liberal win rate: {mixed_lib_rate:.1%}")
    print("  This shows how well the agents can cooperate with different teammates.")

    # DeepRole stats if available
    if hasattr(deeprole_agents[0], 'action_stats'):
        stats = deeprole_agents[0].action_stats
        if stats['total'] > 0:
            print(f"\nDeepRole CFR Statistics:")
            print(f"  Strategies used: {stats['cfr_strategy']}/{stats['total']} ({100*stats['cfr_strategy']/stats['total']:.1f}%)")
            print(f"  Fallbacks: {stats['cfr_fallback']}/{stats['total']} ({100*stats['cfr_fallback']/stats['total']:.1f}%)")
            print(f"  Errors: {stats['cfr_error']}/{stats['total']} ({100*stats['cfr_error']/stats['total']:.1f}%)")


if __name__ == "__main__":
    main()