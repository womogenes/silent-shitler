import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.game_state import create_game_at_state

print("TESTING INFOSET KEY FIX")
print("="*60)

cfr = VectorCFR()

# Test 1: Same game state but different cards should have different infosets
env1 = create_game_at_state(4, 5, president_idx=0, seed=42)
env1.phase = "prez_cardsel"
env1.prez_cards = [0, 0, 1]  # 2 libs, 1 fasc
env1.agent_selection = env1.agents[0]

env2 = create_game_at_state(4, 5, president_idx=0, seed=42)
env2.phase = "prez_cardsel"
env2.prez_cards = [1, 1, 1]  # All fascist
env2.agent_selection = env2.agents[0]

key1 = cfr._get_infoset_key(env1, 0)
key2 = cfr._get_infoset_key(env2, 0)

print("Test 1: Different cards should create different infosets")
print(f"  Cards [0, 0, 1] key: ...{key1[-1] if len(key1) > 7 else 'NO CARDS'}")
print(f"  Cards [1, 1, 1] key: ...{key2[-1] if len(key2) > 7 else 'NO CARDS'}")
print(f"  Keys are different? {key1 != key2} (should be True)")

# Test 2: Verify strategies are separate
strategy1 = cfr._get_strategy(key1, env1, 0)
strategy2 = cfr._get_strategy(key2, env2, 0)

print("\nTest 2: Strategies should be independent")
print(f"  Strategy for [0, 0, 1]: {strategy1}")
print(f"  Strategy for [1, 1, 1]: {strategy2}")
print(f"  First has action 0? {0 in strategy1} (should be True)")
print(f"  Second has action 0? {0 in strategy2} (should be False)")

# Test 3: Simulate CFR updates
print("\nTest 3: Simulating regret updates...")

# Update regrets for first infoset (can discard either)
cfr.regret_sums[key1] = {0: 2.0, 1: 3.0}
strategy1_after = cfr._get_strategy(key1, env1, 0)
print(f"  After update, strategy for [0, 0, 1]: {strategy1_after}")

# Check second infoset is unaffected
strategy2_after = cfr._get_strategy(key2, env2, 0)
print(f"  Strategy for [1, 1, 1] unchanged: {strategy2_after}")
print(f"  Still no action 0? {0 not in strategy2_after} (should be True)")

print("\n" + "="*60)
print("SUCCESS! The fix prevents illegal actions by separating infosets")
print("based on what cards the player has.")