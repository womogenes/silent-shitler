import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.game_state import create_game_at_state
from agents.deeprole.role_assignments import RoleAssignmentManager

print("DEBUGGING ILLEGAL ACTION ATTEMPTS")
print("="*60)

# Create a game state with 4L, 5F (only 2 liberal cards left total)
env = create_game_at_state(4, 5, president_idx=0, seed=42)

# Manually set up a situation where president draws all fascist cards
env.phase = "prez_cardsel"
env.prez_cards = [1, 1, 1]  # All fascist cards
env.president_idx = 0
env.agent_selection = env.agents[0]

print(f"Game state: 4L, 5F")
print(f"President has cards: {env.prez_cards}")
print(f"Phase: {env.phase}")

# Test observation and action mask
obs = env.observe(env.agents[0])
print(f"\nObservation card_action_mask: {obs.get('card_action_mask')}")
print(f"Expected: [0, 1] (can only discard fascist)")

# Test CFR's action extraction
cfr = VectorCFR()
legal_actions = cfr._get_legal_actions(env, 0)
print(f"\nCFR legal actions: {legal_actions}")

# Test strategy generation
infoset_key = cfr._get_infoset_key(env, 0)
print(f"\nInfoset key: {infoset_key}")

# Initialize regret sums if needed
if infoset_key not in cfr.regret_sums:
    cfr.regret_sums[infoset_key] = {0: 0.0, 1: 0.0}
    print(f"Initialized regrets: {cfr.regret_sums[infoset_key]}")

strategy = cfr._get_strategy(infoset_key, env, 0)
print(f"\nStrategy generated: {strategy}")
print(f"Strategy should only contain action 1: {list(strategy.keys()) == [1]}")

# Test what happens if we have mixed regrets
print("\n" + "-"*60)
print("Testing with positive regrets for illegal action:")
cfr.regret_sums[infoset_key] = {0: 5.0, 1: 3.0}  # Positive regret for illegal action 0
print(f"Regrets: {cfr.regret_sums[infoset_key]}")

strategy = cfr._get_strategy(infoset_key, env, 0)
print(f"Strategy: {strategy}")
print(f"Problem: Strategy includes action 0? {0 in strategy}")

if 0 in strategy:
    print("\nFOUND THE BUG: _get_strategy includes illegal actions!")
    print("Even though legal_actions = [1], the strategy includes action 0")
    print("This happens because regrets[0] exists and is positive")