#!/usr/bin/env python
"""Evaluate DeepRole V2 against MetaAgent."""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.eval_agent import evaluate_agents, _AgentFactory
from agents.deeprole.deeprole_agent_v2 import DeepRoleAgentV2
from agents.meta_agent import MetaAgent


def main():
    """Run comprehensive evaluation of DeepRole V2 vs MetaAgent."""
    print("=" * 70)
    print("DEEPROLE V2 vs META AGENT EVALUATION")
    print("=" * 70)

    # Load DeepRole networks
    networks_path = Path("trained_networks_1000_16_15.pkl")
    if not networks_path.exists():
        print(f"Error: Networks not found at {networks_path}")
        return

    print(f"Networks: {networks_path}")
    print("Creating agent pool...")

    # Create agent pools (for non-parallel access)
    deeprole_agents = []
    meta_agents = []

    for _ in range(3):  # Create 3 of each type
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

    # Create agent factories for parallel execution
    deeprole_factory = _AgentFactory(
        DeepRoleAgentV2,
        {
            'networks_path': str(networks_path),
            'cfr_iterations': 15,
            'max_depth': 2
        }
    )
    meta_factory = _AgentFactory(MetaAgent, {'temperature': 1.0})

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
            },
            "factories": [
                deeprole_factory,
                deeprole_factory,
                deeprole_factory,
                meta_factory,
                meta_factory
            ]
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
            },
            "factories": [
                meta_factory,
                meta_factory,
                meta_factory,
                deeprole_factory,
                deeprole_factory
            ]
        },
    ]

    # Run evaluations
    num_games = 200
    all_results = {}

    for config in configs:
        print(f"\n" + "=" * 70)
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 70)
        print(f"Running {num_games} games...")

        # Get agent assignment for this config
        agent_assignment = config["assignment"]()

        # Run evaluation using evaluate_agents with parallel execution
        results = evaluate_agents(
            agents=agent_assignment,
            num_games=num_games,
            seed=42,
            verbose=True,
            track_win_reasons=True,
            num_workers=os.cpu_count() - 4,  # use -1 to enable all cores
            agent_factories=config["factories"]
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
        print("\nDeepRole V2 outperforms MetaAgent overall!")
    elif meta_total > dr_total:
        print("\nMetaAgent outperforms DeepRole V2!")
    else:
        print("\nDeepRole V2 and MetaAgent are evenly matched!")
        print("  Both approaches have their strengths.")

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
