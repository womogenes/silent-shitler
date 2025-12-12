"""
Evaluate Meta Agent against various opponents.

Compares the suspicion-tracking MetaAgent against:
- Random agents
- Selfish agents
- CFR agents (if checkpoint available)
"""

import sys
from pathlib import Path
import time
import pickle
import random
from collections import defaultdict
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from shitler_env.agent import SimpleRandomAgent
from agents.meta_agent import MetaAgent
from agents.selfish_agent import SelfishAgent

# ============================================================================
# Hyperparameters
# ============================================================================

NUM_GAMES = 1000
LOG_EVERY = 100
SEED = 42

ROLE_MAP = {"lib": 0, "fasc": 1, "hitty": 2}


# ============================================================================
# CFR Agent (optional)
# ============================================================================

class CFRAgent:
    """CFR agent that loads from checkpoint."""

    def __init__(self, checkpoint_path=None):
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.iterations = data.get("iterations", 0)

        for k, v in data["regret_sums"].items():
            for a, r in v.items():
                self.regret_sums[k][a] = r

        for k, v in data["strategy_sums"].items():
            for a, s in v.items():
                self.strategy_sums[k][a] = s

        print(f"Loaded CFR: {self.iterations} iterations, {len(self.regret_sums)} infosets")

    def get_average_strategy(self, infoset_key, legal_actions):
        strat_sums = self.strategy_sums[infoset_key]
        total = sum(strat_sums[a] for a in legal_actions)

        if total > 0:
            return {a: strat_sums[a] / total for a in legal_actions}
        else:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def sample_action(self, strategy):
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return random.choices(actions, weights=probs, k=1)[0]


# ============================================================================
# Evaluation Functions
# ============================================================================

def get_legal_actions(obs):
    """Get legal actions from observation masks."""
    for mask_key in ["nomination_mask", "execution_mask", "card_action_mask"]:
        if mask_key in obs:
            return [i for i, v in enumerate(obs[mask_key]) if v == 1]
    return None


def play_game(liberal_agent, fascist_agent, seed=None):
    """Play a single game and return results."""
    env = ShitlerEnv()
    env.reset(seed=seed)

    # Reset meta agent suspicion if applicable
    if hasattr(liberal_agent, 'reset_suspicion'):
        liberal_agent.reset_suspicion()
    if hasattr(fascist_agent, 'reset_suspicion'):
        fascist_agent.reset_suspicion()

    # Assign agents by role
    agents = {}
    for i, agent_name in enumerate(env.agents):
        role = env.roles[agent_name]
        if role == "lib":
            agents[agent_name] = liberal_agent
        else:
            agents[agent_name] = fascist_agent

    step = 0
    while not all(env.terminations.values()) and step < 1000:
        agent_name = env.agent_selection
        agent_idx = env.agents.index(agent_name)
        obs = env.observe(agent_name)
        action_space = env.action_space(agent_name)
        phase = env.phase

        action = agents[agent_name].get_action(
            obs, action_space, phase=phase, agent_idx=agent_idx
        )
        env.step(action)
        step += 1

    # Determine winner
    lib_win = False
    for agent_name, reward in env.rewards.items():
        if reward == 1:
            role = env.roles[agent_name]
            lib_win = (role == "lib")
            break

    # Win condition
    if lib_win:
        win_cond = "lib_5_policies" if env.lib_policies >= 5 else "hitler_executed"
    else:
        win_cond = "fasc_6_policies" if env.fasc_policies >= 6 else "hitler_chancellor"

    return {
        "lib_win": lib_win,
        "win_condition": win_cond,
        "lib_policies": env.lib_policies,
        "fasc_policies": env.fasc_policies,
        "steps": step,
    }


def evaluate_matchup(liberal_agent, fascist_agent, num_games, seed=None, name=""):
    """Evaluate a matchup and return statistics."""
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "win_conditions": defaultdict(int),
        "total_lib_policies": 0,
        "total_fasc_policies": 0,
        "total_steps": 0,
    }

    for i in range(num_games):
        game_seed = seed + i if seed else None
        game_result = play_game(liberal_agent, fascist_agent, seed=game_seed)

        if game_result["lib_win"]:
            results["lib_wins"] += 1
        else:
            results["fasc_wins"] += 1

        results["win_conditions"][game_result["win_condition"]] += 1
        results["total_lib_policies"] += game_result["lib_policies"]
        results["total_fasc_policies"] += game_result["fasc_policies"]
        results["total_steps"] += game_result["steps"]

        if (i + 1) % LOG_EVERY == 0:
            win_rate = 100 * results["lib_wins"] / (i + 1)
            print(f"  [{i+1}/{num_games}] Liberal win rate: {win_rate:.1f}%")

    # Calculate final stats
    results["lib_win_rate"] = results["lib_wins"] / num_games
    results["avg_lib_policies"] = results["total_lib_policies"] / num_games
    results["avg_fasc_policies"] = results["total_fasc_policies"] / num_games
    results["avg_steps"] = results["total_steps"] / num_games
    results["win_conditions"] = dict(results["win_conditions"])
    results["num_games"] = num_games

    return results


def wilson_ci(successes, total, confidence=0.95):
    """Calculate Wilson score confidence interval."""
    from scipy import stats

    if total == 0:
        return 0, 0, 0

    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * ((p * (1 - p) + z**2 / (4 * total)) / total) ** 0.5 / denominator

    return p, max(0, center - margin), min(1, center + margin)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("META AGENT EVALUATION")
    print("=" * 70)
    print(f"Games per matchup: {NUM_GAMES}")
    print(f"Seed: {SEED}")
    print("=" * 70)
    print()

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize agents
    random_agent = SimpleRandomAgent()
    selfish_agent = SelfishAgent()
    meta_agent = MetaAgent(temperature=1.0)

    # Try to load CFR agent
    cfr_checkpoint = Path(__file__).parent.parent / "cfr" / "checkpoints" / "cfr_liberal_iter_350000.pkl"
    cfr_agent = None
    if cfr_checkpoint.exists():
        try:
            # Need infoset module for CFR
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents" / "cfr"))
            from infoset import get_infoset_key
            cfr_agent = CFRAgent(checkpoint_path=str(cfr_checkpoint))
        except Exception as e:
            print(f"Could not load CFR agent: {e}")
            cfr_agent = None

    all_results = {}
    start_time = time.time()

    # ========================================================================
    # Experiment 1: Random vs Random (baseline)
    # ========================================================================
    print("=" * 70)
    print("1. BASELINE: Random Liberals vs Random Fascists")
    print("=" * 70)
    exp_start = time.time()

    results = evaluate_matchup(random_agent, random_agent, NUM_GAMES, SEED, "random_vs_random")
    rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

    print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
    print(f"Win Conditions: {results['win_conditions']}")
    print(f"Avg Policies - Lib: {results['avg_lib_policies']:.1f}, Fasc: {results['avg_fasc_policies']:.1f}")
    print(f"Avg Game Length: {results['avg_steps']:.1f} steps")
    print(f"Time: {time.time() - exp_start:.1f}s")

    all_results["random_vs_random"] = {
        **results,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    print()

    # ========================================================================
    # Experiment 2: Selfish Liberals vs Random Fascists
    # ========================================================================
    print("=" * 70)
    print("2. Selfish Liberals vs Random Fascists")
    print("=" * 70)
    exp_start = time.time()

    results = evaluate_matchup(selfish_agent, random_agent, NUM_GAMES, SEED, "selfish_vs_random")
    rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

    print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
    print(f"Win Conditions: {results['win_conditions']}")
    print(f"Avg Policies - Lib: {results['avg_lib_policies']:.1f}, Fasc: {results['avg_fasc_policies']:.1f}")
    print(f"Time: {time.time() - exp_start:.1f}s")

    all_results["selfish_vs_random"] = {
        **results,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    print()

    # ========================================================================
    # Experiment 3: Meta Liberals vs Random Fascists
    # ========================================================================
    print("=" * 70)
    print("3. META Liberals vs Random Fascists")
    print("=" * 70)
    exp_start = time.time()

    results = evaluate_matchup(meta_agent, random_agent, NUM_GAMES, SEED, "meta_vs_random")
    rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

    print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
    print(f"Win Conditions: {results['win_conditions']}")
    print(f"Avg Policies - Lib: {results['avg_lib_policies']:.1f}, Fasc: {results['avg_fasc_policies']:.1f}")
    print(f"Time: {time.time() - exp_start:.1f}s")

    all_results["meta_vs_random"] = {
        **results,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    print()

    # ========================================================================
    # Experiment 4: Meta Liberals vs Selfish Fascists
    # ========================================================================
    print("=" * 70)
    print("4. META Liberals vs Selfish Fascists")
    print("=" * 70)
    exp_start = time.time()

    results = evaluate_matchup(meta_agent, selfish_agent, NUM_GAMES, SEED, "meta_vs_selfish")
    rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

    print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
    print(f"Win Conditions: {results['win_conditions']}")
    print(f"Avg Policies - Lib: {results['avg_lib_policies']:.1f}, Fasc: {results['avg_fasc_policies']:.1f}")
    print(f"Time: {time.time() - exp_start:.1f}s")

    all_results["meta_vs_selfish"] = {
        **results,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    print()

    # ========================================================================
    # Experiment 5: Selfish vs Selfish (for comparison)
    # ========================================================================
    print("=" * 70)
    print("5. Selfish Liberals vs Selfish Fascists")
    print("=" * 70)
    exp_start = time.time()

    results = evaluate_matchup(selfish_agent, selfish_agent, NUM_GAMES, SEED, "selfish_vs_selfish")
    rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

    print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
    print(f"Win Conditions: {results['win_conditions']}")
    print(f"Time: {time.time() - exp_start:.1f}s")

    all_results["selfish_vs_selfish"] = {
        **results,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    print()

    # ========================================================================
    # Experiment 6: CFR vs Random (if available)
    # ========================================================================
    if cfr_agent:
        print("=" * 70)
        print("6. CFR Liberals vs Random Fascists")
        print("=" * 70)
        exp_start = time.time()

        # Create a wrapper that uses CFR for non-card decisions
        class CFRWrapper:
            def __init__(self, cfr_agent):
                self.cfr = cfr_agent

            def get_action(self, obs, action_space=None, **kwargs):
                phase = kwargs.get("phase", obs.get("phase", None))
                agent_idx = kwargs.get("agent_idx", 0)

                # Hard-code liberal card selection
                if phase in ["prez_cardsel", "chanc_cardsel"]:
                    mask = obs.get("card_action_mask", [1, 1])
                    if mask[1] == 1:
                        return 1
                    return 0

                legal_actions = get_legal_actions(obs)
                if not legal_actions:
                    if action_space:
                        return action_space.sample()
                    return 0

                infoset_key = get_infoset_key(obs, phase, agent_idx)
                strategy = self.cfr.get_average_strategy(infoset_key, legal_actions)
                return self.cfr.sample_action(strategy)

        cfr_wrapper = CFRWrapper(cfr_agent)
        results = evaluate_matchup(cfr_wrapper, random_agent, NUM_GAMES, SEED, "cfr_vs_random")
        rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

        print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
        print(f"Win Conditions: {results['win_conditions']}")
        print(f"Time: {time.time() - exp_start:.1f}s")

        all_results["cfr_vs_random"] = {
            **results,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        print()

        # ====================================================================
        # Experiment 7: Meta vs CFR comparison (Meta lib vs Random, CFR lib vs Random)
        # ====================================================================
        print("=" * 70)
        print("7. CFR Liberals vs Selfish Fascists")
        print("=" * 70)
        exp_start = time.time()

        results = evaluate_matchup(cfr_wrapper, selfish_agent, NUM_GAMES, SEED, "cfr_vs_selfish")
        rate, ci_low, ci_high = wilson_ci(results["lib_wins"], NUM_GAMES)

        print(f"\nLiberal Win Rate: {results['lib_win_rate']*100:.1f}% (95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%)")
        print(f"Win Conditions: {results['win_conditions']}")
        print(f"Time: {time.time() - exp_start:.1f}s")

        all_results["cfr_vs_selfish"] = {
            **results,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        print()

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - start_time

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Matchup':<35} {'Lib Win %':>12} {'95% CI':>20}")
    print("-" * 70)

    for name, res in all_results.items():
        label = name.replace("_", " ").title()
        rate = res["lib_win_rate"] * 100
        ci = f"({res['ci_low']*100:.1f}% - {res['ci_high']*100:.1f}%)"
        print(f"{label:<35} {rate:>11.1f}% {ci:>20}")

    print("-" * 70)
    print()

    # Analysis
    print("ANALYSIS:")
    baseline = all_results["random_vs_random"]["lib_win_rate"]

    for name in ["selfish_vs_random", "meta_vs_random"]:
        if name in all_results:
            rate = all_results[name]["lib_win_rate"]
            diff = (rate - baseline) * 100
            label = "Selfish" if "selfish" in name else "Meta"
            print(f"  {label} improvement over Random baseline: {diff:+.1f}%")

    if "meta_vs_random" in all_results and "selfish_vs_random" in all_results:
        meta_rate = all_results["meta_vs_random"]["lib_win_rate"]
        selfish_rate = all_results["selfish_vs_random"]["lib_win_rate"]
        diff = (meta_rate - selfish_rate) * 100
        if diff > 0:
            print(f"  Meta beats Selfish by: {diff:.1f}%")
        else:
            print(f"  Selfish beats Meta by: {-diff:.1f}%")

    if "cfr_vs_random" in all_results and "meta_vs_random" in all_results:
        cfr_rate = all_results["cfr_vs_random"]["lib_win_rate"]
        meta_rate = all_results["meta_vs_random"]["lib_win_rate"]
        diff = (meta_rate - cfr_rate) * 100
        if diff > 0:
            print(f"  Meta beats CFR by: {diff:.1f}%")
        else:
            print(f"  CFR beats Meta by: {-diff:.1f}%")

    print()
    print(f"Total evaluation time: {total_time/60:.1f} minutes")
    print("=" * 70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"meta_eval_{timestamp}.json"

    output = {
        "metadata": {
            "num_games": NUM_GAMES,
            "seed": SEED,
            "timestamp": timestamp,
            "total_time_seconds": total_time,
        },
        "results": all_results,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
