import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.game_state import create_game_at_state

print("DEBUGGING EMPTY LEGAL ACTIONS CASE")
print("="*60)

cfr = VectorCFR()

# Create game at 4L, 5F
env = create_game_at_state(4, 5, president_idx=0, seed=42)
env.phase = "prez_cardsel"
env.prez_cards = [1, 1, 1]  # All fascist
env.agent_selection = env.agents[0]

# Get legal actions
legal_actions = cfr._get_legal_actions(env, 0)
print(f"Legal actions: {legal_actions}")
print(f"Type: {type(legal_actions)}")
print(f"Is empty? {len(legal_actions) == 0}")

# Test the condition
action = 0  # Illegal action
print(f"\nTesting: action=0 in legal_actions={legal_actions}")
print(f"Result: {action in legal_actions}")

# Now test with no mask
print("\n" + "-"*60)
print("Testing case where no mask is found:")

# Mock environment with no valid mask
class MockEnv:
    def __init__(self):
        self.agents = ["P0"]

env_mock = MockEnv()
obs = {"observation": "something"}  # No mask keys

# Check what _get_legal_actions returns
print(f"Obs with no mask: {obs}")

# Simulate the mask checking logic
legal = []
for mask_key in ['nomination_mask', 'execution_mask', 'card_action_mask']:
    if mask_key in obs:
        print(f"  Found {mask_key}")
        legal = [i for i, valid in enumerate(obs[mask_key]) if valid == 1]
        if legal:
            break

if not legal:
    # This is what happens when no mask is found
    print("  No mask found, returning all actions")
    legal = list(range(2))  # Would return [0, 1]

print(f"Legal actions when no mask: {legal}")
print(f"This would incorrectly include action 0!")

print("\nTHIS IS THE BUG: When no mask is found, _get_legal_actions")
print("returns all actions including illegal ones!")