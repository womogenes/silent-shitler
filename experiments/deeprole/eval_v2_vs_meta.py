#!/usr/bin/env python
"""Evaluate DeepRole V2 against MetaAgent."""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.eval_agent import evaluate_agents, AgentFactory
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

    # Create agent factories for parallel execution
    deeprole_factory = AgentFactory(
        DeepRoleAgentV2,
        {
            'networks_path': str(networks_path),
            'cfr_iterations': 15,
            'max_depth': 2
        }
    )
    meta_factory = AgentFactory(MetaAgent, {'temperature': 1.0})

    # Test configurations
    configs = [
        {
            "name": "3 DeepRole vs 2 MetaAgent",
            "description": "DeepRole as liberals, MetaAgent as fascists",
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
    num_games = os.cpu_count() * 2
    all_results = {}

    for config in configs:
        print(f"\n" + "=" * 70)
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 70)

        # Run evaluation using evaluate_agents with parallel execution
        results = evaluate_agents(
            agent_factories=config["factories"],
            num_games=num_games,
            seed=43,
            track_win_reasons=True,
            num_workers=os.cpu_count(),  # use -1 to enable all cores
            verbose=True,
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


if __name__ == "__main__":
    main()
