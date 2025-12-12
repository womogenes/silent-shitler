import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from shitler_env.game import ShitlerEnv

print("DEBUGGING ACTION MASK EXTRACTION")
print("="*60)

# Create game at 4L, 5F
env = ShitlerEnv()
env.reset(seed=42)
env.lib_policies = 4
env.fasc_policies = 5

# Set up president with all fascist cards
env.phase = "prez_cardsel"
env.prez_cards = [1, 1, 1]  # All fascist
env.president_idx = 0
env.agent_selection = env.agents[0]

print(f"Game state: {env.lib_policies}L, {env.fasc_policies}F")
print(f"President cards: {env.prez_cards}")
print(f"Agent selection: {env.agent_selection}")

# Get observation
obs = env.observe(env.agents[0])
print(f"\ncard_action_mask from obs: {obs.get('card_action_mask')}")

# Extract legal actions the way CFR does
legal_actions = []
for i, valid in enumerate(obs['card_action_mask']):
    if valid == 1:
        legal_actions.append(i)

print(f"Legal actions extracted: {legal_actions}")
print(f"Should be [1] only")

# Test what happens if we try action 0
print(f"\nTrying action 0 (should fail):")
try:
    env.step(0)
    print("  Success?! This shouldn't happen")
except ValueError as e:
    print(f"  Failed as expected: {e}")

# Reset and try action 1
env.prez_cards = [1, 1, 1]
env.phase = "prez_cardsel"
print(f"\nTrying action 1 (should work):")
try:
    env.step(1)
    print("  Success - action 1 worked")
except ValueError as e:
    print(f"  Unexpected failure: {e}")

# Now check if mask values are backwards
print("\n" + "-"*60)
print("Checking if mask interpretation is correct:")
print("card_action_mask[0] = 0 means action 0 is ILLEGAL")
print("card_action_mask[1] = 1 means action 1 is LEGAL")

# Debug: What if the issue is with how we're checking mask values?
print("\nLet me check the exact comparison:")
mask = obs['card_action_mask']
print(f"mask = {mask}")
print(f"mask[0] == 1: {mask[0] == 1}")
print(f"mask[1] == 1: {mask[1] == 1}")

# Check data type
print(f"\nData type of mask: {type(mask)}")
print(f"Data type of mask[0]: {type(mask[0])}")
print(f"mask[0] is exactly 1: {mask[0] == 1}")
print(f"mask[0] is truthy: {bool(mask[0])}")