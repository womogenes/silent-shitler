#!/usr/bin/env python3
"""Quick test to verify all imports work correctly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing imports...")

try:
    from deeprole.role_assignments import RoleAssignmentManager
    print("✓ role_assignments")
except ImportError as e:
    print(f"✗ role_assignments: {e}")

try:
    from deeprole.belief import BeliefTracker
    print("✓ belief")
except ImportError as e:
    print(f"✗ belief: {e}")

try:
    from deeprole.vector_cfr import VectorCFR
    print("✓ vector_cfr")
except ImportError as e:
    print(f"✗ vector_cfr: {e}")

try:
    from deeprole.networks import ValueNetwork, NetworkEnsemble
    print("✓ networks")
except ImportError as e:
    print(f"✗ networks: {e}")

try:
    from deeprole.situation_sampler import AdvancedSituationSampler
    print("✓ situation_sampler")
except ImportError as e:
    print(f"✗ situation_sampler: {e}")

try:
    from deeprole.game_state import create_game_at_state
    print("✓ game_state")
except ImportError as e:
    print(f"✗ game_state: {e}")

try:
    from deeprole.terminal_value_cfr import TerminalValueComputer
    print("✓ terminal_value_cfr")
except ImportError as e:
    print(f"✗ terminal_value_cfr: {e}")

try:
    from deeprole.backwards_training import BackwardsTrainer
    print("✓ backwards_training")
except ImportError as e:
    print(f"✗ backwards_training: {e}")

try:
    from deeprole.gameplay_solver import DeepRoleSolver
    print("✓ gameplay_solver")
except ImportError as e:
    print(f"✗ gameplay_solver: {e}")

print("\nAll imports successful! Ready for training.")
print("\nTo start training with quick test:")
print("  uv run python agents/deeprole/generate_data.py --quick")
print("\nTo verify terminal value diversity:")
print("  uv run python agents/deeprole/verify_terminal_diversity.py")