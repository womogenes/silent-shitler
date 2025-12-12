import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from collections import defaultdict
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.game_state import create_game_at_state

print("DEBUGGING REGRET INITIALIZATION")
print("="*60)

# Create a CFR instance
cfr = VectorCFR()

# Create game at 4L, 5F
env = create_game_at_state(4, 5, president_idx=0, seed=42)
env.phase = "prez_cardsel"
env.prez_cards = [1, 1, 1]  # All fascist
env.agent_selection = env.agents[0]

# Get infoset key
infoset_key = cfr._get_infoset_key(env, 0)
print(f"Infoset key: {infoset_key}")

# Check if regrets exist
print(f"\nRegrets exist? {infoset_key in cfr.regret_sums}")

# Now simulate what happens during CFR iteration
# In _handle_sequential, if infoset is new, it gets initialized
if infoset_key not in cfr.regret_sums:
    # This is what happens implicitly
    cfr.regret_sums[infoset_key] = defaultdict(float)
    print("Initialized empty regret_sums defaultdict")

# Get legal actions
legal_actions = cfr._get_legal_actions(env, 0)
print(f"\nLegal actions: {legal_actions}")

# Get strategy - this is where regrets might get initialized for illegal actions
strategy = cfr._get_strategy(infoset_key, env, 0)
print(f"Initial strategy: {strategy}")

# Now let's simulate adding regrets
print("\n" + "-"*60)
print("Simulating regret updates...")

# This is what _update_regrets_strategies does
action_values = {1: 0.5}  # Only legal action has a value
print(f"Action values: {action_values}")

# But wait, what if action_values accidentally includes illegal actions?
print("\nWhat if action_values incorrectly includes action 0?")
action_values_bad = {0: 0.3, 1: 0.5}
print(f"Bad action values: {action_values_bad}")

# The update would then add regret for action 0
for action in strategy:  # strategy only has legal actions
    print(f"  Updating regret for action {action}")

# But the problem might be that defaultdict creates entries when accessed
print("\n" + "-"*60)
print("Testing defaultdict behavior:")
from collections import defaultdict
test_dict = defaultdict(float)
print(f"Empty defaultdict: {dict(test_dict)}")
print(f"Accessing test_dict[0]: {test_dict[0]}")
print(f"Now dict has: {dict(test_dict)}")

print("\nAHA! defaultdict creates entries when accessed.")
print("If _get_strategy accesses regrets[0] to check its value,")
print("it creates an entry even for illegal actions!")

# Check the _get_strategy implementation
print("\n" + "-"*60)
print("Looking at _get_strategy implementation:")
print("It does: positive_regrets[a] = max(0, regrets.get(a, 0))")
print("Using .get() avoids creating entries, but...")
print("If regrets is a defaultdict and we ever access regrets[a] directly,")
print("it will create the entry.")