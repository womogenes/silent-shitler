#!/usr/bin/env python3
"""Test belief tracker and CFR without neural networks."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.deeprole.deeprole_agent import DeepRoleAgent
from shitler_env.game import ShitlerEnv
import numpy as np


def test_belief_updates():
    """Test that belief tracker updates correctly."""
    agent = DeepRoleAgent(networks_path="nonexistent.pkl", cfr_iterations=5, max_depth=1)
    agent.reset(player_idx=0)

    print("Initial belief (should be uniform):")
    print(f"  Shape: {agent.current_belief.shape}")
    print(f"  Sum: {agent.current_belief.sum():.3f}")
    print(f"  Min: {agent.current_belief.min():.3f}, Max: {agent.current_belief.max():.3f}")

    # Create a mock observation
    obs = {
        'role': 0,  # Liberal
        'lib_policies': 1,
        'fasc_policies': 2,
        'phase': 'nomination',
        'hist_votes': [[1, 0, 1, 0, 1]],  # Some vote history
        'hist_policy': [1],  # One fascist policy played
        'hist_president': [0],
        'hist_chancellor': [1],
        'executed': [0, 0, 0, 0, 0],
    }

    # Update belief
    agent._update_belief(obs)

    print("\nAfter update:")
    print(f"  Shape: {agent.current_belief.shape}")
    print(f"  Sum: {agent.current_belief.sum():.3f}")
    print(f"  Min: {agent.current_belief.min():.3f}, Max: {agent.current_belief.max():.3f}")

    # Check if assignments inconsistent with our role are zeroed out
    role_probs = agent.belief_tracker.get_role_probabilities(agent.current_belief)
    print(f"\nOur role probabilities (we are player 0, liberal):")
    print(f"  Liberal: {role_probs[0]['liberal']:.3f}")
    print(f"  Fascist: {role_probs[0]['fascist']:.3f}")
    print(f"  Hitler: {role_probs[0]['hitler']:.3f}")

    if abs(role_probs[0]['liberal'] - 1.0) < 0.001:
        print("✓ Belief correctly constrains our role to liberal")
    else:
        print("✗ Belief not properly constrained")


def test_cfr_without_networks():
    """Test that CFR runs even without neural networks."""
    agent = DeepRoleAgent(networks_path="nonexistent.pkl", cfr_iterations=5, max_depth=1)

    # Count how many times each action source is used
    for i in range(20):
        obs = {
            'phase': 'nomination',
            'lib_policies': 0,
            'fasc_policies': 0,
            'nomination_mask': [1, 1, 1, 1, 1],  # All players can be nominated
            'president_idx': 0,
        }

        action = agent.get_action(obs, agent_name=f"P{i % 5}")

    print(f"\nAction statistics after 20 decisions:")
    total = agent.action_stats['total']
    if total > 0:
        print(f"  CFR strategy: {agent.action_stats['cfr_strategy']} ({agent.action_stats['cfr_strategy']/total*100:.1f}%)")
        print(f"  CFR fallback: {agent.action_stats['cfr_fallback']} ({agent.action_stats['cfr_fallback']/total*100:.1f}%)")
        print(f"  CFR error: {agent.action_stats['cfr_error']} ({agent.action_stats['cfr_error']/total*100:.1f}%)")

        # Analysis
        if agent.action_stats['cfr_error'] / total > 0.5:
            print("\n✗ CFR is failing most of the time")
        elif agent.action_stats['cfr_fallback'] / total > 0.5:
            print("\n✗ CFR runs but doesn't compute strategies")
        else:
            print("\n✓ CFR is computing strategies successfully")
    else:
        print("  No CFR decisions were made (total = 0)")
        print("\n✗ CFR not being called for these phases")


def main():
    print("=" * 60)
    print("BELIEF TRACKER AND CFR TEST")
    print("=" * 60)

    print("\n1. Testing belief updates...")
    test_belief_updates()

    print("\n" + "-" * 60)
    print("\n2. Testing CFR without networks...")
    test_cfr_without_networks()

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()