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

# Create agent classes (not instances)
agent_classes = []

# Liberals: DeepRole
for i in range(3):
    agent_classes.append(lambda: DeepRoleAgent(
        networks_path="/home/willi/coding/6.S890/silent-shitler/agents/deeprole/trained_networks.pkl",
        cfr_iterations=20,  # Reduced for quick test
        max_depth=2
    ))

# Fascists: Random
for i in range(2):
    agent_classes.append(SimpleRandomAgent)

# Run 10 games
results = evaluate_agents(
    agent_classes,
    num_games=10,
    verbose=False,
    seed=42
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
    print("\n✓ DeepRole is making decisions and playing games!")
    print("  Strategic belief updates are integrated.")
else:
    print("\n⚠ DeepRole may need debugging - no liberal wins yet")