import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from shitler_env.game import ShitlerEnv

print("INVESTIGATING EMPTY MASK BUG")
print("="*60)

# Test various game states to find when mask could be [0, 0]
env = ShitlerEnv()

# Test 1: Normal card selection
print("\n1. Normal card selection (2 libs, 1 fasc):")
env.reset(seed=42)
env.phase = "prez_cardsel"
env.prez_cards = [0, 0, 1]
env.agent_selection = env.agents[0]
obs = env.observe(env.agents[0])
print(f"   Cards: {env.prez_cards}")
print(f"   Mask: {obs.get('card_action_mask')}  (should be [1, 1])")

# Test 2: All fascist cards
print("\n2. All fascist cards:")
env.prez_cards = [1, 1, 1]
obs = env.observe(env.agents[0])
print(f"   Cards: {env.prez_cards}")
print(f"   Mask: {obs.get('card_action_mask')}  (should be [0, 1])")

# Test 3: All liberal cards
print("\n3. All liberal cards:")
env.prez_cards = [0, 0, 0]
obs = env.observe(env.agents[0])
print(f"   Cards: {env.prez_cards}")
print(f"   Mask: {obs.get('card_action_mask')}  (should be [1, 0])")

# Test 4: Empty cards (this shouldn't happen but let's test)
print("\n4. Empty cards (shouldn't happen):")
env.prez_cards = []
obs = env.observe(env.agents[0])
print(f"   Cards: {env.prez_cards}")
print(f"   Mask: {obs.get('card_action_mask')}")

# Test 5: Wrong phase
print("\n5. Wrong phase (not card selection):")
env.phase = "nomination"
obs = env.observe(env.agents[0])
print(f"   Phase: {env.phase}")
print(f"   Has card_action_mask? {'card_action_mask' in obs}")
print(f"   Has nomination_mask? {'nomination_mask' in obs}")

# Test 6: Check what happens in other phases
print("\n6. Voting phase:")
env.phase = "voting"
env.chancellor_nominee = 1
obs = env.observe(env.agents[0])
print(f"   Phase: {env.phase}")
print(f"   Vote mask available? {obs.get('vote') is not None}")

print("\n" + "="*60)
print("CONCLUSION:")
print("If we ever get [0, 0] or empty mask, it's a bug in game.py")
print("There should ALWAYS be at least one legal action")

# Let's check the actual mask generation code
print("\nChecking when card_action_mask could be [0, 0]...")
print("This would require:")
print("  - num_libs == 0 (no liberals to discard)")
print("  - num_fascs == 0 (no fascists to discard)")
print("But if you have 3 cards (or 2 for chancellor), at least one type must exist!")
print("So [0, 0] should be IMPOSSIBLE")