#!/usr/bin/env python3
"""Test belief update implementation."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.deeprole.belief_update import BeliefUpdater
from agents.deeprole.deeprole_agent import DeepRoleAgent
import numpy as np


def test_logical_deductions():
    """Test that logical deductions work correctly."""
    print("Testing logical deductions in belief updates:")
    print("-" * 60)

    updater = BeliefUpdater()

    # Test 1: Own role constraint
    print("\n1. Own role constraint:")
    belief = np.ones(20) / 20
    obs = {
        'role': 0,  # Liberal
        'player_idx': 0
    }

    updated = updater.update_belief(belief, obs)

    # Check that only assignments with P0 as liberal remain
    for i, assignment in enumerate(updater.assignments):
        if assignment[0] == 0:  # P0 is liberal
            assert updated[i] > 0, f"Assignment {i} should have positive probability"
        else:
            assert updated[i] == 0, f"Assignment {i} should have zero probability"

    print("   ✓ Correctly constrains to own role")

    # Test 2: Hitler chancellor deduction
    print("\n2. Hitler chancellor deduction:")
    belief = np.ones(20) / 20
    obs = {
        'fasc_policies': 3,
        'hist_chancellor': [2],  # P2 was chancellor
        'terminations': None  # Game didn't end
    }

    updated = updater.update_belief(belief, obs)

    # Check that assignments with P2 as Hitler are zeroed
    for i, assignment in enumerate(updater.assignments):
        if assignment[2] == 2:  # P2 is Hitler
            assert updated[i] == 0, f"Assignment {i} should be zero (P2 can't be Hitler)"

    print("   ✓ Correctly deduces chancellor is not Hitler")

    # Test 3: Execution deduction
    print("\n3. Execution deduction (Hitler executed):")
    belief = np.ones(20) / 20
    obs = {
        'executed': [0, 0, 1, 0, 0],  # P2 was executed
        'terminations': {'P0': True, 'P1': True, 'P2': True, 'P3': True, 'P4': True}  # Game ended
    }

    updated = updater.update_belief(belief, obs)

    # Check that only assignments with P2 as Hitler remain
    for i, assignment in enumerate(updater.assignments):
        if assignment[2] == 2:  # P2 is Hitler
            assert updated[i] > 0, f"Assignment {i} should have positive probability"
        else:
            assert updated[i] == 0, f"Assignment {i} should be zero (P2 must be Hitler)"

    print("   ✓ Correctly deduces executed player was Hitler")

    print("\n" + "=" * 60)
    print("All logical deduction tests passed!")


def test_belief_in_agent():
    """Test belief updates in DeepRole agent."""
    print("\n" + "=" * 60)
    print("Testing belief updates in DeepRole agent:")
    print("-" * 60)

    # Create agent
    project_root = Path(__file__).parent.parent.parent
    networks_path = project_root / "agents" / "deeprole" / "trained_networks.pkl"

    agent = DeepRoleAgent(networks_path, cfr_iterations=5, max_depth=2)
    agent.reset(player_idx=0)

    print(f"Initial belief shape: {agent.current_belief.shape}")
    print(f"Initial belief sum: {agent.current_belief.sum():.3f}")

    # Give agent an observation with role information
    obs = {
        'role': 0,  # Liberal
        'lib_policies': 0,
        'fasc_policies': 0,
        'phase': 'nomination'
    }

    # Update belief
    agent._update_belief(obs)

    print(f"After update with role=liberal:")
    print(f"  Belief sum: {agent.current_belief.sum():.3f}")

    # Count how many assignments are possible
    possible = np.sum(agent.current_belief > 0)
    print(f"  Possible assignments: {possible}/20")

    # Check that we're liberal in all remaining assignments
    remaining_assignments = []
    for i, prob in enumerate(agent.current_belief):
        if prob > 0:
            assignment = agent.belief_updater.assignments[i]
            assert assignment[0] == 0, f"Player 0 should be liberal in assignment {i}"
            remaining_assignments.append(assignment)

    print(f"  ✓ All remaining assignments have P0 as liberal")

    # Calculate expected number of remaining assignments
    # P0 is liberal, so we need to choose 2 more liberals from 4 remaining players
    # C(4,2) = 6 ways to choose liberals
    # For each, 2 ways to assign fascist/hitler
    # Total: 6 * 2 = 12
    expected = 12
    if possible == expected:
        print(f"  ✓ Correct number of assignments ({expected})")
    else:
        print(f"  ✗ Wrong number of assignments (expected {expected}, got {possible})")


def test_missing_strategies():
    """Test and document the missing strategy component."""
    print("\n" + "=" * 60)
    print("CRITICAL LIMITATION: Missing Strategy Component")
    print("=" * 60)

    print("\nAccording to DeepRole Algorithm 2, belief updates require:")
    print("  b[ρ] = b[ρ] * ∏_i π_i(I_i(h, ρ))")
    print("\nWhere π_i are the strategies computed during CFR.")

    print("\nCurrent implementation:")
    print("  ✓ Logical deductions (zeroing impossible assignments)")
    print("  ✗ Strategy-based updates (requires opponent strategies)")

    print("\nWhy this matters:")
    print("  - Without strategies, we can't do Bayesian inference")
    print("  - We can't learn from voting patterns")
    print("  - We can't infer from policy outcomes")
    print("  - Belief stays nearly uniform except for hard constraints")

    print("\nPossible solutions:")
    print("  1. Store strategies from training (but opponents may differ)")
    print("  2. Assume uniform random opponents (weak assumption)")
    print("  3. Learn opponent models online (complex)")
    print("  4. Use self-play strategies as proxy (biased)")

    print("\nThe paper doesn't fully address this gap between")
    print("training (where we have strategies) and deployment")
    print("(where we don't know opponent strategies).")


def main():
    test_logical_deductions()
    test_belief_in_agent()
    test_missing_strategies()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Belief updates are partially implemented:")
    print("  ✓ Logical deductions work correctly")
    print("  ✗ Strategy-based updates require CFR strategies")
    print("\nThis is a fundamental limitation when playing against")
    print("unknown opponents whose strategies we don't have.")


if __name__ == "__main__":
    main()