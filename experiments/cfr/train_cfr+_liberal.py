"""Train CFR+ agent for Liberal team against random Fascists.

v2 improvements:
- Delayed averaging (skip first d iterations for strategy averaging)
- Sparse data structures (prune zero regrets to save memory)
- Better memory layout (use arrays instead of nested dicts where possible)
"""

import sys
import time
import json
import random
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from agents.cfr.infoset import get_infoset_key

# Wandb logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to console only")

# ============================================================================
# Config
# ============================================================================

USE_WANDB = True
WANDB_PROJECT = "silent-shitler"
WANDB_ENTITY = None

NUM_ITERATIONS = 500000
LOG_EVERY = 1000
EVAL_EVERY = 10000
CHECKPOINT_EVERY = 50000
EVAL_GAMES = 500

# CFR+ specific config
AVERAGING_DELAY = 1000  # Skip first d iterations for strategy averaging
PRUNE_THRESHOLD = 0  # Prune regrets at or below this value (0 = prune zeros)
PRUNE_EVERY = 10000  # How often to prune zero regrets

ROLE_MAP = {"lib": 0, "fasc": 1, "hitty": 2}


class CFRPlusAgentV2:
    """
    CFR+ agent with optimizations:
    - Delayed averaging: don't accumulate strategy until after iteration d
    - Sparse storage: periodically prune zero regrets
    - Linear weighting: w_t = t - d for strategy averaging
    """

    def __init__(self, averaging_delay=1000):
        # infoset_key -> {action: regret_sum}
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        # infoset_key -> {action: strategy_sum}
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
        # Track iteration for linear averaging
        self.iteration = 0
        # Delay before starting to accumulate average strategy
        self.averaging_delay = averaging_delay

    def get_strategy(self, infoset_key, legal_actions):
        """Get current strategy via regret matching."""
        regrets = self.regret_sums[infoset_key]

        # In CFR+, regrets are already non-negative (floored at 0)
        positive_regrets = {a: regrets[a] for a in legal_actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in legal_actions}
        else:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def get_average_strategy(self, infoset_key, legal_actions):
        """Get average strategy (converges to Nash equilibrium)."""
        strat_sums = self.strategy_sums[infoset_key]
        total = sum(strat_sums[a] for a in legal_actions)

        if total > 0:
            return {a: strat_sums[a] / total for a in legal_actions}
        else:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def update_regrets(self, infoset_key, action_regrets):
        """Update regrets with CFR+ floor at 0."""
        for action, regret in action_regrets.items():
            new_regret = self.regret_sums[infoset_key][action] + regret
            # CFR+ floors regrets at 0
            self.regret_sums[infoset_key][action] = max(0, new_regret)

    def update_strategy(self, infoset_key, strategy):
        """Update strategy sum with delayed linear weighting."""
        # Only accumulate after the delay period
        if self.iteration <= self.averaging_delay:
            return

        # Linear weight: w_t = t - d
        weight = self.iteration - self.averaging_delay

        for action, prob in strategy.items():
            self.strategy_sums[infoset_key][action] += weight * prob

    def sample_action(self, strategy):
        """Sample action from strategy distribution."""
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return random.choices(actions, weights=probs, k=1)[0]

    def prune_zero_regrets(self):
        """Remove zero regrets to save memory (CFR+ specific optimization)."""
        pruned_count = 0
        empty_infosets = []

        for infoset_key in list(self.regret_sums.keys()):
            regrets = self.regret_sums[infoset_key]
            # Remove zero entries
            zero_actions = [a for a, r in regrets.items() if r <= PRUNE_THRESHOLD]
            for a in zero_actions:
                del regrets[a]
                pruned_count += 1

            # Mark empty infosets for removal
            if len(regrets) == 0:
                empty_infosets.append(infoset_key)

        # Remove empty infosets from regret_sums (but keep strategy_sums)
        for key in empty_infosets:
            del self.regret_sums[key]

        return pruned_count, len(empty_infosets)

    def get_memory_stats(self):
        """Get memory usage statistics."""
        regret_entries = sum(len(v) for v in self.regret_sums.values())
        strategy_entries = sum(len(v) for v in self.strategy_sums.values())
        return {
            "regret_infosets": len(self.regret_sums),
            "strategy_infosets": len(self.strategy_sums),
            "regret_entries": regret_entries,
            "strategy_entries": strategy_entries,
        }


def get_legal_actions(env, agent):
    """Get legal actions for agent in current state."""
    obs = env.observe(agent)
    action_space = env.action_space(agent)

    if obs is None:
        return []

    if "nomination_mask" in obs:
        return [i for i, v in enumerate(obs["nomination_mask"]) if v == 1]
    elif "execution_mask" in obs:
        return [i for i, v in enumerate(obs["execution_mask"]) if v == 1]
    elif "card_action_mask" in obs:
        return [i for i, v in enumerate(obs["card_action_mask"]) if v == 1]
    else:
        return list(range(action_space.n))


def get_roles_list(env):
    """Get numeric roles for all players."""
    return [ROLE_MAP[env.roles[f"P{i}"]] for i in range(5)]


def cfr_liberal_iteration(env, cfr_agent, traverser_idx):
    """
    CFR+ iteration where only liberals use CFR, fascists play randomly.
    """
    roles = get_roles_list(env)
    liberal_players = {i for i, r in enumerate(roles) if r == 0}

    # Only train if traverser is a liberal
    if traverser_idx not in liberal_players:
        # Just play out the game randomly
        while not all(env.terminations.values()):
            agent = env.agent_selection
            legal_actions = get_legal_actions(env, agent)
            if legal_actions:
                env.step(random.choice(legal_actions))
            else:
                env.step(0)
        return 0.0

    return _cfr_traverse(env, cfr_agent, traverser_idx, liberal_players, 1.0, 1.0, 1.0)


def _cfr_traverse(env, cfr_agent, traverser_idx, liberal_players, pi_i, pi_neg_i, pi_sample):
    """Recursive CFR+ traversal."""
    if all(env.terminations.values()):
        traverser_agent = env.agents[traverser_idx]
        return env.rewards[traverser_agent], 1.0

    current_agent = env.agent_selection
    current_idx = env.agents.index(current_agent)
    obs = env.observe(current_agent)
    phase = env.phase

    legal_actions = get_legal_actions(env, current_agent)
    if not legal_actions:
        env.step(0)
        return _cfr_traverse(env, cfr_agent, traverser_idx, liberal_players, pi_i, pi_neg_i, pi_sample)

    # Fascists play randomly
    if current_idx not in liberal_players:
        action = random.choice(legal_actions)
        env.step(action)
        return _cfr_traverse(env, cfr_agent, traverser_idx, liberal_players, pi_i, pi_neg_i, pi_sample)

    # Liberal player - use CFR (except for hard-coded decisions)

    # HARD-CODE: Liberals always discard fascist if possible
    if phase in ["prez_cardsel", "chanc_cardsel"] and 1 in legal_actions:
        action = 1  # discard fascist
        env.step(action)
        return _cfr_traverse(env, cfr_agent, traverser_idx, liberal_players, pi_i, pi_neg_i, pi_sample)

    infoset_key = get_infoset_key(obs, phase, current_idx)
    strategy = cfr_agent.get_strategy(infoset_key, legal_actions)

    # Sample action
    action = cfr_agent.sample_action(strategy)
    action_prob = strategy[action]

    env.step(action)

    if current_idx == traverser_idx:
        utility, tail = _cfr_traverse(
            env, cfr_agent, traverser_idx, liberal_players,
            pi_i * action_prob, pi_neg_i, pi_sample * action_prob
        )

        # Importance-sampled counterfactual value
        W = utility * pi_neg_i / pi_sample if pi_sample > 0 else 0

        # Compute regrets
        action_regrets = {}
        for a in legal_actions:
            if a == action:
                action_regrets[a] = W * (1 - action_prob)
            else:
                action_regrets[a] = -W * action_prob

        # CFR+ update (floors at 0)
        cfr_agent.update_regrets(infoset_key, action_regrets)

        # Update strategy sum (with delayed linear weighting)
        cfr_agent.update_strategy(infoset_key, strategy)

        return utility, tail * action_prob
    else:
        # Other liberal (not traverser)
        utility, tail = _cfr_traverse(
            env, cfr_agent, traverser_idx, liberal_players,
            pi_i, pi_neg_i * action_prob, pi_sample * action_prob
        )
        return utility, tail * action_prob


def evaluate_vs_random(cfr_agent, num_games):
    """Evaluate CFR liberals against random fascists."""
    results = {
        "liberal_wins": 0,
        "fascist_wins": 0,
        "win_conditions": defaultdict(int),
    }

    for _ in range(num_games):
        env = ShitlerEnv()
        env.reset()
        roles = get_roles_list(env)
        liberal_players = {i for i, r in enumerate(roles) if r == 0}

        while not all(env.terminations.values()):
            agent = env.agent_selection
            idx = env.agents.index(agent)
            obs = env.observe(agent)
            phase = env.phase

            legal_actions = get_legal_actions(env, agent)
            if not legal_actions:
                env.step(0)
                continue

            if idx in liberal_players:
                # CFR liberal (with hard-coded card selection)
                if phase in ["prez_cardsel", "chanc_cardsel"] and 1 in legal_actions:
                    action = 1  # always discard fascist
                else:
                    infoset_key = get_infoset_key(obs, phase, idx)
                    strategy = cfr_agent.get_average_strategy(infoset_key, legal_actions)
                    action = cfr_agent.sample_action(strategy)
            else:
                # Random fascist
                action = random.choice(legal_actions)

            env.step(action)

        # Determine winner
        lib_idx = roles.index(0)
        lib_reward = env.rewards[f"P{lib_idx}"]

        if lib_reward > 0:
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

    return results


def count_infoset_stats(cfr_agent):
    """Count infosets by phase and role."""
    stats = {
        "total": len(cfr_agent.regret_sums),
        "by_phase": defaultdict(int),
        "by_role": defaultdict(int),
        "visited": 0,
    }

    for key in cfr_agent.regret_sums.keys():
        if len(key) >= 5:
            role = key[0]
            phase = key[4]
            stats["by_phase"][phase] += 1
            stats["by_role"][role] += 1

        if key in cfr_agent.strategy_sums:
            total_strat = sum(cfr_agent.strategy_sums[key].values())
            if total_strat > 0:
                stats["visited"] += 1

    return stats


def main():
    print("=" * 70)
    print("CFR+ LIBERAL TRAINING v2 (vs Random Fascists)")
    print("  - Delayed averaging (d={})".format(AVERAGING_DELAY))
    print("  - Sparse regret pruning (every {} iters)".format(PRUNE_EVERY))
    print("=" * 70)

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    config = {
        "algorithm": "cfr_plus_v2_liberal_only",
        "num_iterations": NUM_ITERATIONS,
        "log_every": LOG_EVERY,
        "eval_every": EVAL_EVERY,
        "eval_games": EVAL_GAMES,
        "averaging_delay": AVERAGING_DELAY,
        "prune_every": PRUNE_EVERY,
        "opponent": "random_fascist",
    }

    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    print()

    # Initialize wandb
    use_wandb = USE_WANDB and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=config,
            name=f"cfr_liberal_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        print(f"Wandb logging enabled: {wandb.run.url}")
    else:
        print("Wandb logging disabled")
    print()

    cfr_agent = CFRPlusAgentV2(averaging_delay=AVERAGING_DELAY)
    print("Note: Liberal card selection is HARD-CODED (always discard fascist)")
    print(f"Note: Strategy averaging starts after iteration {AVERAGING_DELAY}")
    print()

    start_time = time.time()
    last_log_time = start_time
    last_infosets = len(cfr_agent.regret_sums)

    for iteration in range(1, NUM_ITERATIONS + 1):
        cfr_agent.iteration = iteration

        # Create fresh game
        env = ShitlerEnv()
        env.reset()

        # Alternate traverser across all 5 players
        traverser_idx = iteration % 5

        # Run CFR+ iteration
        cfr_liberal_iteration(env, cfr_agent, traverser_idx)

        # Periodic pruning of zero regrets
        if iteration % PRUNE_EVERY == 0:
            pruned, empty = cfr_agent.prune_zero_regrets()
            if pruned > 0:
                print(f"  [Pruned {pruned} zero regrets, {empty} empty infosets]")

        # Periodic logging
        if iteration % LOG_EVERY == 0:
            now = time.time()
            elapsed = now - start_time
            batch_time = now - last_log_time
            last_log_time = now

            mem_stats = cfr_agent.get_memory_stats()
            num_infosets = mem_stats["regret_infosets"]
            new_infosets = num_infosets - last_infosets
            last_infosets = num_infosets

            iters_per_sec = LOG_EVERY / batch_time

            stats = count_infoset_stats(cfr_agent)
            coverage = stats["visited"] / stats["total"] * 100 if stats["total"] > 0 else 0

            # Show if we're in warmup period
            warmup_str = " [warmup]" if iteration <= AVERAGING_DELAY else ""

            print(f"[{iteration:>7}/{NUM_ITERATIONS}]{warmup_str} "
                  f"infosets={num_infosets:>7} (+{new_infosets:>5}) | "
                  f"visited={stats['visited']:>7} ({coverage:.1f}%) | "
                  f"{iters_per_sec:.0f} it/s | "
                  f"elapsed={elapsed:.0f}s")

            if use_wandb:
                wandb.log({
                    "iteration": iteration,
                    "infosets": num_infosets,
                    "new_infosets": new_infosets,
                    "visited_infosets": stats["visited"],
                    "coverage_pct": coverage,
                    "iters_per_sec": iters_per_sec,
                    "elapsed_seconds": elapsed,
                    "regret_entries": mem_stats["regret_entries"],
                    "strategy_entries": mem_stats["strategy_entries"],
                    "in_warmup": iteration <= AVERAGING_DELAY,
                }, step=iteration)

        # Periodic evaluation
        if iteration % EVAL_EVERY == 0:
            print(f"\n--- Evaluation at iteration {iteration} ---")

            eval_start = time.time()
            results = evaluate_vs_random(cfr_agent, EVAL_GAMES)
            eval_time = time.time() - eval_start

            lib_win_rate = results["liberal_wins"] / EVAL_GAMES

            print(f"Liberal Win Rate: {lib_win_rate*100:.1f}%")
            print(f"Win conditions: {dict(results['win_conditions'])}")
            print(f"Evaluation took {eval_time:.1f}s")

            if use_wandb:
                wandb.log({
                    "eval/liberal_win_rate": lib_win_rate,
                    "eval/fascist_win_rate": results["fascist_wins"] / EVAL_GAMES,
                    "eval/lib_5_policies": results["win_conditions"]["lib_5_policies"],
                    "eval/fasc_6_policies": results["win_conditions"]["fasc_6_policies"],
                    "eval/hitler_chancellor": results["win_conditions"]["hitler_chancellor"],
                    "eval/hitler_executed": results["win_conditions"]["hitler_executed"],
                }, step=iteration)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"cfr_liberal_v2_{iteration}_{timestamp}.json"

            results_with_meta = {
                "agent_type": "cfr_plus_v2_liberal",
                "iterations": iteration,
                "infosets": len(cfr_agent.regret_sums),
                "training_time_seconds": time.time() - start_time,
                "timestamp": timestamp,
                "liberal_win_rate": lib_win_rate,
                "averaging_delay": AVERAGING_DELAY,
                **results
            }

            with open(results_file, "w") as f:
                json.dump(results_with_meta, f, indent=2)
            print(f"Results saved to {results_file}\n")

        # Periodic checkpointing
        if iteration % CHECKPOINT_EVERY == 0:
            path = checkpoint_dir / f"cfr_liberal_v2_iter_{iteration}.pkl"
            data = {
                "iterations": iteration,
                "averaging_delay": AVERAGING_DELAY,
                "regret_sums": dict(cfr_agent.regret_sums),
                "strategy_sums": dict(cfr_agent.strategy_sums),
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved checkpoint to {path}")

    # Final save
    total_time = time.time() - start_time
    path = checkpoint_dir / f"cfr_liberal_v2_iter_{NUM_ITERATIONS}.pkl"
    data = {
        "iterations": NUM_ITERATIONS,
        "averaging_delay": AVERAGING_DELAY,
        "regret_sums": dict(cfr_agent.regret_sums),
        "strategy_sums": dict(cfr_agent.strategy_sums),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

    print("=" * 70)
    print("Training complete!")
    print(f"Total iterations: {NUM_ITERATIONS}")
    print(f"Total infosets: {len(cfr_agent.regret_sums)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
