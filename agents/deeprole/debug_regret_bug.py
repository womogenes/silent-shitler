import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from collections import defaultdict
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.game_state import create_game_at_state

print("DEBUGGING REGRET INITIALIZATION BUG")
print("="*60)

cfr = VectorCFR()

# Create game at 4L, 5F with all fascist cards
env = create_game_at_state(4, 5, president_idx=0, seed=42)
env.phase = "prez_cardsel"
env.prez_cards = [1, 1, 1]  # All fascist
env.agent_selection = env.agents[0]

print(f"Game state: 4L, 5F")
print(f"President cards: {env.prez_cards} (all fascist)")
print(f"Legal actions should be: [1] only")

# Get the infoset key
infoset_key = cfr._get_infoset_key(env, 0)
print(f"\nInfoset key: {infoset_key}")

# Check initial state
print(f"\nInitial regret_sums has this infoset? {infoset_key in cfr.regret_sums}")
print(f"Initial strategy_sums has this infoset? {infoset_key in cfr.strategy_sums}")

# First time getting strategy - this initializes regrets
strategy1 = cfr._get_strategy(infoset_key, env, 0)
print(f"\nFirst strategy: {strategy1}")
print(f"Now regret_sums has infoset? {infoset_key in cfr.regret_sums}")

# Check what's in regret_sums
if infoset_key in cfr.regret_sums:
    print(f"Regret_sums content: {dict(cfr.regret_sums[infoset_key])}")

print("\n" + "-"*60)
print("Simulating what happens when regrets get updated...")

# Simulate an update (this is what _update_regrets_strategies does)
# Let's say we only computed value for action 1
action_values = {1: 0.5}
strategy = {1: 1.0}
reach_prob = np.ones(20) * 0.05  # uniform

# Expected value
ev = sum(strategy[a] * action_values[a] for a in strategy)
print(f"\nAction values: {action_values}")
print(f"Strategy: {strategy}")
print(f"Expected value: {ev}")

# Update regrets
reach_weight = np.sum(reach_prob)
for action in strategy:
    regret = action_values[action] - ev
    # This is where regret_sums gets populated
    if infoset_key not in cfr.regret_sums:
        cfr.regret_sums[infoset_key] = defaultdict(float)

    cfr.regret_sums[infoset_key][action] = max(
        cfr.regret_sums[infoset_key][action] + regret * reach_weight, 0
    )

print(f"\nAfter update, regret_sums: {dict(cfr.regret_sums[infoset_key])}")

# Now what if we somehow had action 0 in action_values?
print("\n" + "-"*60)
print("What if action_values incorrectly included action 0?")

# This could happen if _handle_sequential computed values for illegal actions
action_values_bad = {0: 0.3, 1: 0.5}
print(f"Bad action_values: {action_values_bad}")

# The update loop only iterates over actions in strategy (which are legal)
print("But update only processes actions in strategy, so action 0 wouldn't be added")
print("Unless... strategy incorrectly included action 0")

print("\n" + "="*60)
print("CONCLUSION: The bug must be that strategy is including illegal actions")
print("This could happen if regret_sums already has entries for illegal actions")
print("from a previous iteration with different cards")