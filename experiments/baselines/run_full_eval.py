import sys
from pathlib import Path
import random
import json
import math
import pickle
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from agents.random_agent import RandomAgent
from agents.selfish_agent import SelfishAgent
from agents.meta_agent import MetaAgent

# Try to import PPO
try:
    import torch
    from agents.ppo.ppo_agent import PPOAgent
    from agents.ppo.observation import ObservationProcessor
    from agents.ppo.networks import PHASE_TO_IDX
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: PPO not available (torch not installed?)")

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

NUM_GAMES = 1000
SEED = 42

# Best meta agent config from sweep
BEST_META_CONFIG = {
    "fasc_policy_prez_sus": 2.5,
    "fasc_policy_chanc_sus": 2.0,
    "conflict_sus": 1.5,
    "vote_threshold_mult": 1.0,
}


def wilson_ci(wins, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return 0, 0, 0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - spread), min(1, center + spread)


def chi_square_test(wins1, n1, wins2, n2):
    """Chi-square test for difference in proportions."""
    table = [[wins1, n1 - wins1], [wins2, n2 - wins2]]
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    return chi2, p_value


class ConfigurableMetaAgent(MetaAgent):
    """Meta agent with configurable parameters."""

    def __init__(self, config):
        super().__init__()
        self.fasc_policy_prez_sus = config.get("fasc_policy_prez_sus", 2.0)
        self.fasc_policy_chanc_sus = config.get("fasc_policy_chanc_sus", 1.0)
        self.conflict_sus = config.get("conflict_sus", 1.5)
        self.vote_threshold_mult = config.get("vote_threshold_mult", 1.2)

    def _update_suspicion(self, obs):
        """Update suspicion with configurable values."""
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

            if policy == 1:
                self.suspicion[prez] += self.fasc_policy_prez_sus
                self.suspicion[chanc] += self.fasc_policy_chanc_sus

            if prez_claim >= 0 and chanc_claim >= 0:
                if self._detect_conflict(prez_claim, chanc_claim, policy):
                    if (prez, chanc) not in self.detected_conflicts:
                        self.detected_conflicts.add((prez, chanc))
                        self.suspicion[prez] += self.conflict_sus
                        self.suspicion[chanc] += self.conflict_sus

        self.last_processed_govt = len(hist_succeeded) - 1


class CFRAgentWrapper:
    """Wrapper for CFR agent loaded from checkpoint."""

    def __init__(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        self.regret_sums = defaultdict(lambda: defaultdict(float), data["regret_sums"])
        self.strategy_sums = defaultdict(lambda: defaultdict(float), data["strategy_sums"])

        # Import infoset key function (same as notebook)
        from agents.cfr.infoset import get_infoset_key
        self.get_infoset_key = get_infoset_key

    def get_average_strategy(self, infoset_key, legal_actions):
        strat_sums = self.strategy_sums[infoset_key]
        total = sum(strat_sums[a] for a in legal_actions)
        if total > 0:
            return {a: strat_sums[a] / total for a in legal_actions}
        else:
            return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def get_action(self, obs, action_space=None, **kwargs):
        phase = kwargs.get("phase", "voting")
        agent_idx = kwargs.get("agent_idx", 0)

        # Hard-coded: liberals always discard fascist
        if phase in ["prez_cardsel", "chanc_cardsel"]:
            mask = obs.get("card_action_mask", [1, 1])
            return 1 if mask[1] == 1 else 0

        # Get legal actions
        if "nomination_mask" in obs:
            legal_actions = [i for i, v in enumerate(obs["nomination_mask"]) if v == 1]
        elif "execution_mask" in obs:
            legal_actions = [i for i, v in enumerate(obs["execution_mask"]) if v == 1]
        elif "card_action_mask" in obs:
            legal_actions = [i for i, v in enumerate(obs["card_action_mask"]) if v == 1]
        else:
            legal_actions = list(range(action_space.n if action_space else 2))

        if not legal_actions:
            return 0

        infoset_key = self.get_infoset_key(obs, phase, agent_idx)
        strategy = self.get_average_strategy(infoset_key, legal_actions)

        # Sample from strategy
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return random.choices(actions, weights=probs, k=1)[0]


class PPOAgentWrapper:
    """Wrapper for PPO agent loaded from checkpoint."""

    def __init__(self, checkpoint_path):
        self.obs_processor = ObservationProcessor()
        obs_dim = self.obs_processor.obs_dim

        self.ppo_agent = PPOAgent(obs_dim)
        self.ppo_agent.load(checkpoint_path)
        self.ppo_agent.policy.eval()

    def get_action(self, obs, action_space=None, **kwargs):
        phase = kwargs.get("phase", "voting")

        # Hard-coded: liberals always discard fascist (consistent with other agents)
        if phase in ["prez_cardsel", "chanc_cardsel"]:
            mask = obs.get("card_action_mask", [1, 1])
            return 1 if mask[1] == 1 else 0

        # Process observation
        obs_array = self.obs_processor.process(obs)
        phase_idx = PHASE_TO_IDX.get(phase, 0)

        # Get action mask
        if phase == "nomination":
            action_mask = np.array(obs.get("nomination_mask", [0] * 5), dtype=np.float32)
        elif phase == "execution":
            action_mask = np.array(obs.get("execution_mask", [0] * 5), dtype=np.float32)
        elif phase == "voting":
            action_mask = np.ones(2, dtype=np.float32)
        elif phase == "prez_claim":
            action_mask = np.ones(4, dtype=np.float32)
        elif phase == "chanc_claim":
            action_mask = np.ones(3, dtype=np.float32)
        else:
            action_mask = np.ones(2, dtype=np.float32)

        action, _, _ = self.ppo_agent.get_action(obs_array, phase_idx, action_mask, deterministic=True)
        return action


def run_matchup(lib_agent_factory, fasc_agent_factory, num_games, seed):
    """Run games with liberal agent vs fascist agent."""
    random.seed(seed)
    np.random.seed(seed)

    results = {
        "liberal_wins": 0,
        "fascist_wins": 0,
        "win_conditions": defaultdict(int),
    }

    for i in tqdm(range(num_games), desc="Games", leave=False):
        env = ShitlerEnv()
        env.reset(seed=seed + i)

        # Create fresh agents
        lib_agent = lib_agent_factory()
        fasc_agent = fasc_agent_factory()

        # Reset if needed
        if hasattr(lib_agent, "reset_suspicion"):
            lib_agent.reset_suspicion()

        # Map agents to roles
        agents = {}
        for agent_name in env.agents:
            role = env.roles[agent_name]
            if role == "lib":
                agents[agent_name] = lib_agent
            else:
                agents[agent_name] = fasc_agent

        while not all(env.terminations.values()):
            agent_name = env.agent_selection
            obs = env.observe(agent_name)
            action_space = env.action_space(agent_name)
            agent_idx = env.agents.index(agent_name)

            action = agents[agent_name].get_action(
                obs, action_space, phase=env.phase, agent_idx=agent_idx
            )
            env.step(action)

        # Record result
        for name, reward in env.rewards.items():
            if reward == 1:
                role = env.roles[name]
                if role == "lib":
                    results["liberal_wins"] += 1
                    if env.lib_policies >= 5:
                        results["win_conditions"]["lib_5_policies"] += 1
                    else:
                        results["win_conditions"]["hitler_executed"] += 1
                else:
                    results["fascist_wins"] += 1
                    if env.fasc_policies >= 6:
                        results["win_conditions"]["fasc_6_policies"] += 1
                    else:
                        results["win_conditions"]["hitler_chancellor"] += 1
                break

    return results


def main():
    print("=" * 70)
    print("FULL AGENT EVALUATION FOR NEURIPS")
    print("=" * 70)
    print(f"Games per matchup: {NUM_GAMES}")
    print(f"Seed: {SEED}")
    print()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Define agents
    agents = {
        "Random": lambda: RandomAgent(),
        "Selfish": lambda: SelfishAgent(),
        "Meta": lambda: ConfigurableMetaAgent(BEST_META_CONFIG),
    }

    # Try to load CFR checkpoint (same as notebook) - load ONCE
    cfr_checkpoint = Path(__file__).parent.parent / "cfr" / "checkpoints" / "cfr_liberal_iter_350000.pkl"

    if cfr_checkpoint.exists():
        print(f"Loading CFR from {cfr_checkpoint}")
        cfr_agent_instance = CFRAgentWrapper(cfr_checkpoint)
        agents["CFR"] = lambda a=cfr_agent_instance: a
    else:
        print("CFR checkpoint not found, skipping CFR agent")

    # Try to load PPO checkpoint - load ONCE
    ppo_checkpoint = Path(__file__).parent.parent / "checkpoints_asymmetric" / "liberal" / "checkpoint_1000.pt"
    if not ppo_checkpoint.exists():
        ppo_checkpoint = Path(__file__).parent.parent / "checkpoints_asymmetric" / "liberal" / "best_model.pt"

    if ppo_checkpoint.exists() and PPO_AVAILABLE:
        print(f"Loading PPO from {ppo_checkpoint}")
        ppo_agent_instance = PPOAgentWrapper(ppo_checkpoint)
        agents["PPO"] = lambda a=ppo_agent_instance: a
    else:
        if not PPO_AVAILABLE:
            print("PPO not available (torch not installed)")
        else:
            print("PPO checkpoint not found, skipping PPO agent")

    # Opponent types
    opponents = {
        "Random": lambda: RandomAgent(),
        "Selfish": lambda: SelfishAgent(),
    }

    # Run all matchups
    all_results = {}

    for agent_name, agent_factory in agents.items():
        for opp_name, opp_factory in opponents.items():
            matchup = f"{agent_name}_vs_{opp_name}"
            print(f"Running {matchup}...")

            results = run_matchup(agent_factory, opp_factory, NUM_GAMES, SEED)

            # Compute CI
            wins = results["liberal_wins"]
            p, ci_low, ci_high = wilson_ci(wins, NUM_GAMES)

            results["win_rate"] = p
            results["ci_low"] = ci_low
            results["ci_high"] = ci_high
            results["num_games"] = NUM_GAMES

            all_results[matchup] = results

            print(f"  Liberal win rate: {p:.1%} [{ci_low:.1%}, {ci_high:.1%}]")

    # Compute significance tests
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Chi-square tests)")
    print("=" * 70)

    significance_tests = {}
    agent_names = list(agents.keys())

    for opp_name in opponents.keys():
        print(f"\nvs {opp_name} opponents:")

        # Sort agents by win rate
        sorted_agents = sorted(
            agent_names,
            key=lambda a: all_results[f"{a}_vs_{opp_name}"]["win_rate"],
            reverse=True
        )

        for i in range(len(sorted_agents) - 1):
            a1, a2 = sorted_agents[i], sorted_agents[i + 1]
            r1 = all_results[f"{a1}_vs_{opp_name}"]
            r2 = all_results[f"{a2}_vs_{opp_name}"]

            chi2, p_val = chi_square_test(
                r1["liberal_wins"], NUM_GAMES,
                r2["liberal_wins"], NUM_GAMES
            )

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {a1} vs {a2}: chi2={chi2:.2f}, p={p_val:.4f} {sig}")

            significance_tests[f"{a1}_vs_{a2}_{opp_name}"] = {
                "chi2": float(chi2),
                "p_value": float(p_val),
                "significant_05": bool(p_val < 0.05),
                "significant_01": bool(p_val < 0.01),
            }

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Agent':<10} {'vs Random':<25} {'vs Selfish':<25}")
    print("-" * 60)

    for agent_name in agent_names:
        r_rand = all_results.get(f"{agent_name}_vs_Random", {})
        r_self = all_results.get(f"{agent_name}_vs_Selfish", {})

        rand_str = f"{r_rand.get('win_rate', 0):.1%} [{r_rand.get('ci_low', 0):.1%}, {r_rand.get('ci_high', 0):.1%}]" if r_rand else "-"
        self_str = f"{r_self.get('win_rate', 0):.1%} [{r_self.get('ci_low', 0):.1%}, {r_self.get('ci_high', 0):.1%}]" if r_self else "-"

        print(f"{agent_name:<10} {rand_str:<25} {self_str:<25}")

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Bar chart with error bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, opp_name in enumerate(["Random", "Selfish"]):
        ax = axes[idx]

        names = []
        win_rates = []
        ci_lows = []
        ci_highs = []

        for agent_name in agent_names:
            r = all_results.get(f"{agent_name}_vs_{opp_name}", {})
            if r:
                names.append(agent_name)
                win_rates.append(r["win_rate"] * 100)
                ci_lows.append((r["win_rate"] - r["ci_low"]) * 100)
                ci_highs.append((r["ci_high"] - r["win_rate"]) * 100)

        x = np.arange(len(names))
        bars = ax.bar(x, win_rates, yerr=[ci_lows, ci_highs], capsize=5)

        ax.set_xlabel("Liberal Agent")
        ax.set_ylabel("Liberal Win Rate (%)")
        ax.set_title(f"vs {opp_name} Fascists")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.axhline(y=50, linestyle="--", color="gray", alpha=0.5)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(results_dir / "agent_comparison.png", dpi=150)
    plt.savefig(results_dir / "agent_comparison.pdf")
    print(f"  Saved agent_comparison.png/pdf")

    # Plot 2: Win conditions breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    win_cond_labels = ["5 Lib Policies", "Hitler Executed", "6 Fasc Policies", "Hitler Chancellor"]
    win_cond_keys = ["lib_5_policies", "hitler_executed", "fasc_6_policies", "hitler_chancellor"]

    for idx, opp_name in enumerate(["Random", "Selfish"]):
        ax = axes[idx]

        data = {k: [] for k in win_cond_keys}
        names = []

        for agent_name in agent_names:
            r = all_results.get(f"{agent_name}_vs_{opp_name}", {})
            if r:
                names.append(agent_name)
                wc = r.get("win_conditions", {})
                for k in win_cond_keys:
                    data[k].append(wc.get(k, 0) / NUM_GAMES * 100)

        x = np.arange(len(names))
        width = 0.2

        for i, (key, label) in enumerate(zip(win_cond_keys, win_cond_labels)):
            ax.bar(x + i * width, data[key], width, label=label)

        ax.set_xlabel("Liberal Agent")
        ax.set_ylabel("Percentage of Games (%)")
        ax.set_title(f"Win Conditions vs {opp_name} Fascists")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(names)
        ax.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "win_conditions.png", dpi=150)
    plt.savefig(results_dir / "win_conditions.pdf")
    print(f"  Saved win_conditions.png/pdf")

    # Save all results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "num_games": NUM_GAMES,
            "seed": SEED,
            "timestamp": timestamp,
            "agents": list(agents.keys()),
            "opponents": list(opponents.keys()),
            "meta_config": BEST_META_CONFIG,
        },
        "results": {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_results.items()},
        "significance_tests": significance_tests,
    }

    # Convert defaultdicts in win_conditions
    for k, v in output["results"].items():
        if "win_conditions" in v:
            v["win_conditions"] = dict(v["win_conditions"])

    output_file = results_dir / f"full_eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
