"""Evaluation script to test DeepRole with strategic belief updates."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import DeepRoleAgent


def test_strategic_deeprole(num_games=100):
    """Test DeepRole with strategic belief updates.

    Configuration: 3 liberal DeepRole vs 2 random fascist/Hitler
    """
    print("=" * 60)
    print("DEEPROLE WITH STRATEGIC BELIEF UPDATES")
    print("=" * 60)
    print(f"Testing {num_games} games")
    print("Configuration: 3 liberal DeepRole vs 2 random fascist/Hitler")
    print("-" * 60)

    # Create role-specific agents
    lib_agents = []
    fasc_agents = []

    # Liberals: DeepRole with strategic belief updates
    for _ in range(3):
        agent = DeepRoleAgent(
            networks_path="/home/willi/coding/6.S890/silent-shitler/agents/deeprole/trained_networks.pkl",
            cfr_iterations=50,  # Real-time CFR iterations
            max_depth=3
        )
        lib_agents.append(agent)

    # Fascists and Hitler: Random agents
    for _ in range(2):
        agent = SimpleRandomAgent()
        fasc_agents.append(agent)

    # Run evaluation
    results = evaluate_agents(
        lib_agents=lib_agents,
        fasc_agents=fasc_agents,
        num_games=num_games,
        progress_bar=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    lib_wins = results['liberal_wins']
    fasc_wins = results['fascist_wins']
    lib_win_rate = lib_wins / num_games

    print(f"Liberal wins: {lib_wins}/{num_games} ({lib_win_rate:.1%})")
    print(f"Fascist wins: {fasc_wins}/{num_games} ({(1-lib_win_rate):.1%})")

    if 'win_reasons' in results:
        print("\nWin reasons:")
        for reason, count in results['win_reasons'].items():
            print(f"  {reason}: {count}")

    print("\n" + "=" * 60)

    # Analysis
    if lib_win_rate < 0.4:
        print("⚠️  DeepRole performance below expectations")
        print("Possible reasons:")
        print("- Networks may need more training")
        print("- Strategic belief updates may have bugs")
        print("- CFR parameters may need tuning")
    elif lib_win_rate > 0.6:
        print("✓ DeepRole with strategic belief updates working well!")
        print("  The agent is effectively using CFR-computed strategies")
        print("  for belief updates and decision making.")
    else:
        print("→ DeepRole performing moderately")
        print("  This is expected with partially trained networks")

    return results


if __name__ == "__main__":
    # Quick test first
    print("\nRunning quick test (10 games)...")
    print("=" * 60)
    quick_results = test_strategic_deeprole(10)

    # Full evaluation if quick test passes
    if quick_results['liberal_wins'] > 0:
        print("\n\nRunning full evaluation (100 games)...")
        print("=" * 60)
        full_results = test_strategic_deeprole(100)
    else:
        print("\n⚠️ Quick test shows issues - skipping full evaluation")
        print("Check implementation and debug output")