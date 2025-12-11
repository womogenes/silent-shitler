"""
Simplified CFR+ training for liberals with coarser state abstraction.
Uses only 3 suspicion buckets per player to improve coverage.
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
from agents.cfr.infoset_simple import (
    get_infoset_key_simple,
    calculate_state_space_size,
    get_simple_history_abstraction
)

# ============================================================================
# Hyperparameters
# ============================================================================

NUM_ITERATIONS = 500000
LOG_EVERY = 5000
CHECKPOINT_EVERY = 50000
EVAL_EVERY = 10000
EVAL_GAMES = 500

ROLE_MAP = {"lib": 0, "fasc": 1, "hitty": 2}


# ============================================================================
# Simplified CFR+ Agent
# ============================================================================

class SimpleCFRPlusAgent:
    """CFR+ agent with simplified state abstraction."""

    def __init__(self):
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
        self.iteration = 0

    def get_strategy(self, infoset_key, legal_actions):
        """Get current strategy using regret matching."""
        regrets = self.regret_sums[infoset_key]

        # Regret matching
        positive_regrets = {a: max(0, regrets[a]) for a in legal_actions}
        total = sum(positive_regrets.values())

        if total > 0:
            strategy = {a: positive_regrets[a] / total for a in legal_actions}
        else:
            # Uniform if no positive regrets
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        return strategy

    def get_average_strategy(self, infoset_key, legal_actions):
        """Get average strategy over all iterations."""
        strategy_sum = self.strategy_sums[infoset_key]
        total = sum(strategy_sum[a] for a in legal_actions)

        if total > 0:
            return {a: strategy_sum[a] / total for a in legal_actions}
        else:
            return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def update_regrets(self, infoset_key, action_regrets):
        """Update regrets with CFR+ floor at 0."""
        for action, regret in action_regrets.items():
            self.regret_sums[infoset_key][action] += regret
            # CFR+ floors regrets at 0
            if self.regret_sums[infoset_key][action] < 0:
                self.regret_sums[infoset_key][action] = 0

    def update_strategy(self, infoset_key, strategy, weight=1.0):
        """Update strategy sum with linear weighting (CFR+ style)."""
        for action, prob in strategy.items():
            self.strategy_sums[infoset_key][action] += weight * prob

    def sample_action(self, strategy):
        """Sample action from strategy distribution."""
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return random.choices(actions, weights=probs, k=1)[0]


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
    Uses simplified infoset abstraction for better coverage.
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
    """Recursive CFR+ traversal with simplified abstraction."""
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
        action_prob = 1.0
        env.step(action)
        return _cfr_traverse(env, cfr_agent, traverser_idx, liberal_players, pi_i, pi_neg_i, pi_sample)

    # Use simplified infoset key
    infoset_key = get_infoset_key_simple(obs, phase, current_idx)
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

        # Update strategy sum with linear weighting
        cfr_agent.update_strategy(infoset_key, strategy, weight=cfr_agent.iteration)

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
                    infoset_key = get_infoset_key_simple(obs, phase, idx)
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
        "visited": 0,  # infosets with non-zero strategy sum
    }

    for key in cfr_agent.regret_sums.keys():
        if len(key) >= 5:
            role = key[0]
            phase = key[4]
            stats["by_phase"][phase] += 1
            stats["by_role"][role] += 1

        # Check if visited (has strategy sum)
        if key in cfr_agent.strategy_sums:
            total_strat = sum(cfr_agent.strategy_sums[key].values())
            if total_strat > 0:
                stats["visited"] += 1

    return stats


def main():
    print("=" * 70)
    print("SIMPLIFIED CFR+ LIBERAL TRAINING (vs Random Fascists)")
    print("=" * 70)

    # Print state space analysis
    sizes = calculate_state_space_size()
    print(f"State space size: {sizes['total_states']:,} states")
    print(f"  Liberal perspective: {sizes['liberal_states']:,} states")
    print(f"  Fascist perspective: {sizes['fascist_states']:,} states")
    print(f"  Suspicion buckets: 3 per player (3^5 = {sizes['suspicion_combinations']} combinations)")
    print(f"  Reduction from original: ~3,253x smaller")
    print("=" * 70)
    print()

    checkpoint_dir = Path(__file__).parent / "checkpoints_simple"
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    config = {
        "algorithm": "cfr_plus_liberal_simple",
        "num_iterations": NUM_ITERATIONS,
        "state_space_size": sizes['total_states'],
        "liberal_states": sizes['liberal_states'],
        "suspicion_buckets": 3,
        "log_every": LOG_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
        "eval_every": EVAL_EVERY,
        "eval_games": EVAL_GAMES,
        "timestamp": datetime.now().isoformat(),
    }

    # Create agent
    cfr_agent = SimpleCFRPlusAgent()

    # Training loop
    start_time = time.time()

    for i in range(1, NUM_ITERATIONS + 1):
        cfr_agent.iteration = i

        # Create fresh game
        env = ShitlerEnv()
        env.reset()

        # Alternate traverser across all 5 players
        traverser_idx = i % 5

        # Run CFR iteration
        cfr_liberal_iteration(env, cfr_agent, traverser_idx)

        # Logging
        if i % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            its_per_sec = i / elapsed
            stats = count_infoset_stats(cfr_agent)
            coverage = 100 * stats['total'] / sizes['liberal_states']

            print(f"[Iteration {i:,}/{NUM_ITERATIONS:,}]")
            print(f"  Infosets: {stats['total']:,} ({coverage:.1f}% of liberal state space)")
            print(f"  Visited: {stats['visited']:,}")
            print(f"  Speed: {its_per_sec:.0f} iterations/sec")
            print(f"  Time: {elapsed:.1f}s")
            print()

        # Evaluation
        if i % EVAL_EVERY == 0:
            print(f"--- Evaluation at iteration {i} ---")
            eval_start = time.time()
            results = evaluate_vs_random(cfr_agent, EVAL_GAMES)
            eval_time = time.time() - eval_start

            win_rate = 100 * results['liberal_wins'] / EVAL_GAMES
            stats = count_infoset_stats(cfr_agent)
            coverage = 100 * stats['total'] / sizes['liberal_states']

            print(f"Liberal Win Rate: {win_rate:.1f}%")
            print(f"State Coverage: {stats['total']:,}/{sizes['liberal_states']:,} ({coverage:.1f}%)")
            print(f"Win conditions: {dict(results['win_conditions'])}")
            print(f"Evaluation took {eval_time:.1f}s")

            # Save evaluation results
            eval_results = {
                "iteration": i,
                "win_rate": win_rate,
                "liberal_wins": results['liberal_wins'],
                "fascist_wins": results['fascist_wins'],
                "win_conditions": dict(results['win_conditions']),
                "infosets": stats['total'],
                "coverage_percent": coverage,
                "eval_time": eval_time,
            }

            results_file = results_dir / f"cfr_simple_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, "w") as f:
                json.dump(eval_results, f, indent=2)

            print(f"Results saved to {results_file}")
            print()

        # Checkpointing
        if i % CHECKPOINT_EVERY == 0:
            checkpoint_path = checkpoint_dir / f"cfr_simple_iter_{i}.pkl"
            data = {
                "iterations": i,
                "regret_sums": dict(cfr_agent.regret_sums),
                "strategy_sums": dict(cfr_agent.strategy_sums),
                "config": config,
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(data, f)

            stats = count_infoset_stats(cfr_agent)
            coverage = 100 * stats['total'] / sizes['liberal_states']
            print(f"Saved checkpoint to {checkpoint_path}")
            print(f"  Total infosets: {stats['total']:,} ({coverage:.1f}% coverage)")
            print()

    # Final stats
    total_time = time.time() - start_time
    final_stats = count_infoset_stats(cfr_agent)
    coverage = 100 * final_stats['total'] / sizes['liberal_states']

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Iterations: {NUM_ITERATIONS:,}")
    print(f"Final infosets: {final_stats['total']:,}")
    print(f"State space coverage: {coverage:.1f}%")
    print(f"By phase: {dict(final_stats['by_phase'])}")
    print("=" * 70)

    # Save final model
    final_path = checkpoint_dir / f"cfr_simple_final_{NUM_ITERATIONS}.pkl"
    data = {
        "iterations": NUM_ITERATIONS,
        "regret_sums": dict(cfr_agent.regret_sums),
        "strategy_sums": dict(cfr_agent.strategy_sums),
        "config": config,
        "final_stats": final_stats,
        "training_time": total_time,
    }
    with open(final_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()