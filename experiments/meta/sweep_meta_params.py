"""
Parameter sweep for MetaAgent suspicion scores.

Sweeps over:
- fasc_policy_prez_sus: Suspicion added to president when fascist policy played
- fasc_policy_chanc_sus: Suspicion added to chancellor when fascist policy played
- conflict_sus: Suspicion added to both players when claim conflict detected
- vote_threshold_mult: Multiplier for voting threshold (higher = stricter)
"""

import sys
from pathlib import Path
import json
import itertools
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from shitler_env.agent import SimpleRandomAgent, BaseAgent
from agents.selfish_agent import SelfishAgent

import random
import math
from tqdm import tqdm


# ============================================================================
# Configurable MetaAgent
# ============================================================================

class ConfigurableMetaAgent(BaseAgent):
    """MetaAgent with configurable suspicion parameters."""

    def __init__(
        self,
        fasc_policy_prez_sus=2.0,
        fasc_policy_chanc_sus=1.0,
        conflict_sus=1.5,
        vote_threshold_mult=1.2,
        temperature=1.0,
    ):
        self.fasc_policy_prez_sus = fasc_policy_prez_sus
        self.fasc_policy_chanc_sus = fasc_policy_chanc_sus
        self.conflict_sus = conflict_sus
        self.vote_threshold_mult = vote_threshold_mult
        self.temperature = temperature
        self.reset_suspicion()

    def reset_suspicion(self):
        self.suspicion = [1.0] * 5
        self.last_processed_govt = -1
        self.detected_conflicts = set()

    def get_action(self, obs, action_space=None, **kwargs):
        phase = kwargs.get("phase", obs.get("phase", None))
        self._update_suspicion(obs)

        if phase == "nomination" or "nomination_mask" in obs:
            return self._handle_nomination(obs)
        elif phase == "voting":
            return self._handle_voting(obs)
        elif phase in ["prez_cardsel", "chanc_cardsel"] or "card_action_mask" in obs:
            return self._handle_card_selection(obs)
        elif phase == "prez_claim":
            return self._handle_prez_claim(obs)
        elif phase == "chanc_claim":
            return self._handle_chanc_claim(obs)
        elif phase == "execution" or "execution_mask" in obs:
            return self._handle_execution(obs)

        valid_actions = self.get_valid_actions(obs)
        if valid_actions:
            return random.choice(valid_actions)
        if action_space:
            return action_space.sample()
        return 0

    def _update_suspicion(self, obs):
        hist_succeeded = obs.get("hist_succeeded", [])
        hist_policy = obs.get("hist_policy", [])
        hist_president = obs.get("hist_president", [])
        hist_chancellor = obs.get("hist_chancellor", [])
        hist_prez_claim = obs.get("hist_prez_claim", [])
        hist_chanc_claim = obs.get("hist_chanc_claim", [])

        for i in range(self.last_processed_govt + 1, len(hist_succeeded)):
            if hist_succeeded[i] != 1:
                continue

            prez = hist_president[i]
            chanc = hist_chancellor[i]
            policy = hist_policy[i]
            prez_claim = hist_prez_claim[i]
            chanc_claim = hist_chanc_claim[i]

            # Fascist policy - add suspicion
            if policy == 1:
                self.suspicion[prez] += self.fasc_policy_prez_sus
                self.suspicion[chanc] += self.fasc_policy_chanc_sus

            # Conflict detection
            if prez_claim >= 0 and chanc_claim >= 0:
                if self._detect_conflict(prez_claim, chanc_claim, policy):
                    if (prez, chanc) not in self.detected_conflicts:
                        self.detected_conflicts.add((prez, chanc))
                        self.suspicion[prez] += self.conflict_sus
                        self.suspicion[chanc] += self.conflict_sus

        self.last_processed_govt = len(hist_succeeded) - 1

    def _detect_conflict(self, prez_claim, chanc_claim, policy_played):
        max_libs_to_chanc = min(prez_claim, 2)
        if chanc_claim > max_libs_to_chanc:
            return True
        if policy_played == 0 and chanc_claim == 0:
            return True
        return False

    def _handle_nomination(self, obs):
        mask = obs.get("nomination_mask", [])
        valid = [i for i, v in enumerate(mask) if v == 1]
        if not valid:
            return 0
        scores = [(i, self.suspicion[i]) for i in valid]
        scores.sort(key=lambda x: x[1])
        return scores[0][0]

    def _handle_voting(self, obs):
        prez_idx = obs.get("president_idx", 0)
        chanc_idx = obs.get("chancellor_nominee", -1)
        if chanc_idx < 0:
            return 1

        govt_suspicion = (self.suspicion[prez_idx] + self.suspicion[chanc_idx]) / 2
        executed = obs.get("executed", [0] * 5)
        alive_suspicion = [self.suspicion[i] for i in range(5) if executed[i] == 0]
        avg_suspicion = sum(alive_suspicion) / len(alive_suspicion) if alive_suspicion else 1.0

        threshold = avg_suspicion * self.vote_threshold_mult

        if govt_suspicion < threshold:
            return 1
        else:
            diff = govt_suspicion - threshold
            prob_yes = max(0.1, 1.0 / (1.0 + diff))
            return 1 if random.random() < prob_yes else 0

    def _handle_card_selection(self, obs):
        mask = obs.get("card_action_mask", [1, 1])
        cards = obs.get("cards", [])
        num_libs = 0
        for i, v in enumerate(cards):
            if v == 1:
                num_libs = i
                break
        total_cards = 3 if len(cards) == 4 else 2
        num_fascs = total_cards - num_libs
        valid = [i for i, v in enumerate(mask) if v == 1]

        if 1 in valid and num_fascs > 0:
            return 1
        if 0 in valid and num_libs > 0:
            return 0
        return random.choice(valid) if valid else 0

    def _handle_prez_claim(self, obs):
        personal_cards = obs.get("personal_cards_seen", [])
        if personal_cards:
            last_entry = personal_cards[-1]
            if isinstance(last_entry, (list, tuple)) and len(last_entry) >= 2:
                return last_entry[1]
        return 1

    def _handle_chanc_claim(self, obs):
        personal_cards = obs.get("personal_cards_seen", [])
        if personal_cards:
            last_entry = personal_cards[-1]
            if isinstance(last_entry, (list, tuple)) and len(last_entry) >= 2:
                return min(last_entry[1], 2)
        return 1

    def _handle_execution(self, obs):
        mask = obs.get("execution_mask", [])
        valid = [i for i, v in enumerate(mask) if v == 1]
        if not valid:
            return 0
        scores = [(i, self.suspicion[i]) for i in valid]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]


# ============================================================================
# Evaluation
# ============================================================================

def play_game(liberal_agent, fascist_agent, seed=None):
    env = ShitlerEnv()
    env.reset(seed=seed)

    if hasattr(liberal_agent, 'reset_suspicion'):
        liberal_agent.reset_suspicion()

    agents = {}
    for agent_name in env.agents:
        role = env.roles[agent_name]
        if role == "lib":
            agents[agent_name] = liberal_agent
        else:
            agents[agent_name] = fascist_agent

    step = 0
    while not all(env.terminations.values()) and step < 1000:
        agent_name = env.agent_selection
        obs = env.observe(agent_name)
        action_space = env.action_space(agent_name)
        phase = env.phase
        agent_idx = env.agents.index(agent_name)

        action = agents[agent_name].get_action(
            obs, action_space, phase=phase, agent_idx=agent_idx
        )
        env.step(action)
        step += 1

    for agent_name, reward in env.rewards.items():
        if reward == 1:
            return env.roles[agent_name] == "lib"
    return False


def evaluate_config(config, num_games=500, seed=42, opponent="random"):
    agent = ConfigurableMetaAgent(**config)

    if opponent == "random":
        opp_agent = SimpleRandomAgent()
    else:
        opp_agent = SelfishAgent()

    wins = 0
    for i in range(num_games):
        if play_game(agent, opp_agent, seed=seed + i):
            wins += 1

    return wins / num_games


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("META AGENT PARAMETER SWEEP")
    print("=" * 70)

    # Parameter grid
    param_grid = {
        "fasc_policy_prez_sus": [1.0, 1.5, 2.0, 2.5, 3.0],
        "fasc_policy_chanc_sus": [0.5, 1.0, 1.5, 2.0],
        "conflict_sus": [0.5, 1.0, 1.5, 2.0, 2.5],
        "vote_threshold_mult": [1.0, 1.1, 1.2, 1.3, 1.5],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Total configurations: {len(combinations)}")
    print(f"Parameters: {keys}")
    print()

    NUM_GAMES = 500
    SEED = 42

    results = []

    # Evaluate each configuration
    for i, combo in enumerate(tqdm(combinations, desc="Sweeping")):
        config = dict(zip(keys, combo))

        # Evaluate vs Random
        win_rate_random = evaluate_config(config, NUM_GAMES, SEED, "random")

        # Evaluate vs Selfish
        win_rate_selfish = evaluate_config(config, NUM_GAMES, SEED + 10000, "selfish")

        results.append({
            "config": config,
            "win_rate_random": win_rate_random,
            "win_rate_selfish": win_rate_selfish,
            "combined_score": win_rate_random * 0.5 + win_rate_selfish * 0.5,
        })

        # Print progress every 50 configs
        if (i + 1) % 50 == 0:
            best_so_far = max(results, key=lambda x: x["combined_score"])
            print(f"\n[{i+1}/{len(combinations)}] Best so far:")
            print(f"  Config: {best_so_far['config']}")
            print(f"  vs Random: {best_so_far['win_rate_random']:.1%}")
            print(f"  vs Selfish: {best_so_far['win_rate_selfish']:.1%}")
            print(f"  Combined: {best_so_far['combined_score']:.1%}")

    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    # Print top 10
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)

    for i, r in enumerate(results[:10]):
        print(f"\n#{i+1}:")
        print(f"  Config: {r['config']}")
        print(f"  vs Random: {r['win_rate_random']:.1%}")
        print(f"  vs Selfish: {r['win_rate_selfish']:.1%}")
        print(f"  Combined: {r['combined_score']:.1%}")

    # Print best for each opponent type
    print("\n" + "=" * 70)
    print("BEST FOR EACH OPPONENT")
    print("=" * 70)

    best_random = max(results, key=lambda x: x["win_rate_random"])
    print(f"\nBest vs Random:")
    print(f"  Config: {best_random['config']}")
    print(f"  Win rate: {best_random['win_rate_random']:.1%}")

    best_selfish = max(results, key=lambda x: x["win_rate_selfish"])
    print(f"\nBest vs Selfish:")
    print(f"  Config: {best_selfish['config']}")
    print(f"  Win rate: {best_selfish['win_rate_selfish']:.1%}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "num_games": NUM_GAMES,
            "seed": SEED,
            "param_grid": param_grid,
            "total_configs": len(combinations),
            "timestamp": timestamp,
        },
        "top_10": results[:10],
        "best_vs_random": best_random,
        "best_vs_selfish": best_selfish,
        "all_results": results,
    }

    output_file = results_dir / f"param_sweep_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print the ultimate best config
    print("\n" + "=" * 70)
    print("BEST META AGENT CONFIG")
    print("=" * 70)
    best = results[0]
    print(f"fasc_policy_prez_sus = {best['config']['fasc_policy_prez_sus']}")
    print(f"fasc_policy_chanc_sus = {best['config']['fasc_policy_chanc_sus']}")
    print(f"conflict_sus = {best['config']['conflict_sus']}")
    print(f"vote_threshold_mult = {best['config']['vote_threshold_mult']}")
    print(f"\nCombined win rate: {best['combined_score']:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
