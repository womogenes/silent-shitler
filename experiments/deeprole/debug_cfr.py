"""Debug CFR issues in DeepRole agent."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import traceback
from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import DeepRoleAgent


def debug_single_game():
    """Run a single game with debugging output."""
    print("=" * 60)
    print("DEBUG SINGLE GAME")
    print("=" * 60)

    # Create a single DeepRole agent
    agent = DeepRoleAgent(
        networks_path="/home/willi/coding/6.S890/silent-shitler/agents/deeprole/trained_networks.pkl",
        cfr_iterations=10,  # Reduced for debugging
        max_depth=2
    )

    # Create agent classes
    agent_classes = [
        lambda: agent,  # DeepRole as P0
        SimpleRandomAgent,
        SimpleRandomAgent,
        SimpleRandomAgent,
        SimpleRandomAgent
    ]

    # Run one game
    try:
        results = evaluate_agents(
            agent_classes,
            num_games=1,
            verbose=True,
            seed=42
        )
        print("\nGame completed successfully!")
        print(f"Result: {'Liberal' if results['lib_wins'] > 0 else 'Fascist'} win")

    except Exception as e:
        print(f"\nGame failed with error: {e}")
        traceback.print_exc()


def test_cfr_directly():
    """Test CFR solver directly to identify the issue."""
    print("\n" + "=" * 60)
    print("TESTING CFR DIRECTLY")
    print("=" * 60)

    from agents.deeprole.vector_cfr import VectorCFR
    from agents.deeprole.game_state import create_game_at_state

    # Create CFR solver
    cfr = VectorCFR()

    # Create a simple game state
    env = create_game_at_state(0, 0, 0)

    # Create uniform belief
    belief = np.ones(20) / 20

    print("Running CFR solve_situation...")
    try:
        values = cfr.solve_situation(
            env,
            belief,
            num_iterations=10,
            averaging_delay=3,
            neural_nets=None,
            max_depth=2
        )
        print(f"CFR succeeded! Values shape: {np.array(values).shape if values is not None else None}")

        # Try to extract strategies
        strategies = cfr.get_average_strategies()
        print(f"Extracted {len(strategies)} strategies")

    except Exception as e:
        print(f"CFR failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # First test CFR directly
    test_cfr_directly()

    # Then try a single game
    print("\n")
    debug_single_game()