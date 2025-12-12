"""Test script to verify agent interfaces work correctly."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))

from agent import SimpleRandomAgent
from eval_agent import evaluate_agents

# Import wrapped agents
from ppo import PPOAgent
from cfr import CFRAgent
from random_agent import RandomAgent


def test_agents():
    """Test that all agents work with the standard interface."""
    print("Testing agent interfaces...")

    # Test 1: Random agents (both old and new)
    print("\n1. Testing RandomAgent compatibility...")
    agents = [RandomAgent for _ in range(5)]
    results = evaluate_agents(agents, num_games=2, verbose=False, seed=42)
    print(f"   Random agents - Liberal win rate: {results['lib_win_rate']:.0%}")

    # Test 2: SimpleRandomAgent
    print("\n2. Testing SimpleRandomAgent...")
    agents = [SimpleRandomAgent for _ in range(5)]
    results = evaluate_agents(agents, num_games=2, verbose=False, seed=42)
    print(f"   Simple random agents - Liberal win rate: {results['lib_win_rate']:.0%}")

    # Test 3: PPO agent (with default initialization)
    print("\n3. Testing PPO agent wrapper...")
    try:
        agents = [PPOAgent for _ in range(5)]
        results = evaluate_agents(agents, num_games=1, verbose=False, seed=42)
        print(f"   PPO agents initialized successfully")
    except Exception as e:
        print(f"   PPO agent test failed: {e}")

    # Test 4: CFR agent
    print("\n4. Testing CFR agent wrapper...")
    try:
        agents = [CFRAgent for _ in range(5)]
        results = evaluate_agents(agents, num_games=1, verbose=False, seed=42)
        print(f"   CFR agents initialized successfully")
    except Exception as e:
        print(f"   CFR agent test failed: {e}")

    # Test 5: Mixed agents
    print("\n5. Testing mixed agent types...")
    try:
        agents = [RandomAgent, SimpleRandomAgent, PPOAgent, CFRAgent, RandomAgent]
        results = evaluate_agents(agents, num_games=1, verbose=False, seed=42)
        print(f"   Mixed agents work together successfully")
    except Exception as e:
        print(f"   Mixed agent test failed: {e}")

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_agents()