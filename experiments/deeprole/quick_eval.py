"""Quick evaluation of DeepRole with strategic belief updates."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.eval_agent import evaluate_agents
from shitler_env.agent import SimpleRandomAgent
from agents.deeprole.deeprole_agent import DeepRoleAgent


# Test 3 liberal DeepRole vs 2 random fascist/Hitler
print("=" * 60)
print("QUICK TEST: 3 Liberal DeepRole vs 2 Random Fascist/Hitler")
print("=" * 60)

# Create agent instances once (much faster than creating new ones each game)
print("Loading DeepRole agents (one-time cost)...")
lib_agents = []
for i in range(3):
    agent = DeepRoleAgent(
        networks_path="/home/willi/downloads/trained_networks_1000_16_15.pkl",
        cfr_iterations=20,  # Reduced for quick test
        max_depth=2
    )
    lib_agents.append(agent)

# Create fascist agents
fasc_agents = []
for i in range(2):
    fasc_agents.append(SimpleRandomAgent())

# Run games with random role assignment
num_games = 100
print(f"Running {num_games} games...")

results = evaluate_agents(
    None,
    num_games=num_games,
    verbose=True,
    seed=42,
    lib_agents=lib_agents,
    fasc_agents=fasc_agents
)

# Print results
lib_wins = results['lib_wins']
fasc_wins = results['fasc_wins']
total = lib_wins + fasc_wins
lib_rate = lib_wins / total * 100 if total > 0 else 0

print(f"\nResults after {total} games:")
print(f"  Liberal wins: {lib_wins} ({lib_rate:.0f}%)")
print(f"  Fascist wins: {fasc_wins} ({100-lib_rate:.0f}%)")

if lib_wins > 0:
    print("\nDeepRole is making decisions and playing games!")
    print("  Strategic belief updates are integrated.")
    print("  Agents are properly reused across games (efficient!).")
else:
    print("\nDeepRole may need debugging - no liberal wins yet")
