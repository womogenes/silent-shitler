"""Evaluate Selfish Agent vs CFR Agent.

Compares selfish algorithm against trained CFR+ liberal agent.
"""

import sys
import json
import pickle
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shitler_env"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents" / "cfr"))

from game import ShitlerEnv
from agent import SimpleRandomAgent, BaseAgent
from infoset import get_infoset_key


class SelfishAgent(BaseAgent):
    """
    Selfish agent that prioritizes its own party's policies.
    """

    def get_action(self, obs, action_space=None, **kwargs):
        role = obs.get("role", 0)
        is_liberal = (role == 0)

        valid_actions = self.get_valid_actions(obs)

        if "card_action_mask" in obs:
            return self._selfish_card_action(obs, is_liberal)

        if valid_actions:
            return random.choice(valid_actions)

        if action_space:
            return action_space.sample()

        return 0

    def _selfish_card_action(self, obs, is_liberal):
        cards = obs.get("cards", [])
        mask = obs.get("card_action_mask", [1, 1])

        num_libs = 0
        for i, v in enumerate(cards):
            if v == 1:
                num_libs = i
                break

        total_cards = 3 if len(cards) == 4 else 2
        num_fascs = total_cards - num_libs

        valid = [i for i, v in enumerate(mask) if v == 1]

        if is_liberal:
            if 1 in valid and num_fascs > 0:
                return 1
            if 0 in valid and num_libs > 0:
                return 0
        else:
            if 0 in valid and num_libs > 0:
                return 0
            if 1 in valid and num_fascs > 0:
                return 1

        return random.choice(valid) if valid else 0


class CFRAgent(BaseAgent):
    """
    CFR+ agent that uses trained strategy tables.

    Only plays as liberal - uses average strategy from training.
    For card selection, hard-codes liberal play (always discard fascist).
    """

    def __init__(self, checkpoint_path=None):
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path):
        """Load trained CFR agent from checkpoint."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.iterations = data.get("iterations", 0)

        # Reconstruct defaultdicts
        for k, v in data["regret_sums"].items():
            for a, r in v.items():
                self.regret_sums[k][a] = r

        for k, v in data["strategy_sums"].items():
            for a, s in v.items():
                self.strategy_sums[k][a] = s

        print(f"Loaded CFR checkpoint: {self.iterations} iterations, {len(self.regret_sums)} infosets")

    def get_average_strategy(self, infoset_key, legal_actions):
        """Get average strategy (converges to Nash)."""
        strat_sums = self.strategy_sums[infoset_key]
        total = sum(strat_sums[a] for a in legal_actions)

        if total > 0:
            return {a: strat_sums[a] / total for a in legal_actions}
        else:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def sample_action(self, strategy):
        """Sample action from strategy distribution."""
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return random.choices(actions, weights=probs, k=1)[0]

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action using trained CFR strategy."""
        phase = kwargs.get("phase", None)

        # Infer phase from observation
        if phase is None:
            if "nomination_mask" in obs:
                phase = "nomination"
            elif "execution_mask" in obs:
                phase = "execution"
            elif "card_action_mask" in obs:
                # Could be prez or chanc
                cards = obs.get("cards", [])
                phase = "prez_cardsel" if len(cards) == 4 else "chanc_cardsel"
            else:
                phase = "voting"

        # Hard-code: liberals always discard fascist if possible
        if phase in ["prez_cardsel", "chanc_cardsel"]:
            mask = obs.get("card_action_mask", [1, 1])
            if mask[1] == 1:  # Can discard fascist
                return 1
            return 0

        # Get legal actions
        legal_actions = self.get_valid_actions(obs)
        if not legal_actions:
            if action_space:
                return action_space.sample()
            return 0

        # Get infoset key
        agent_idx = kwargs.get("agent_idx", obs.get("president_idx", 0))
        infoset_key = get_infoset_key(obs, phase, agent_idx)

        # Get strategy and sample
        strategy = self.get_average_strategy(infoset_key, legal_actions)
        return self.sample_action(strategy)


def evaluate_role_based_agents(liberal_agent, fascist_agent,
                                num_games=100, seed=None, verbose=True):
    """
    Evaluate with different agent instances for different roles.
    """
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "win_conditions": defaultdict(int),
    }

    for game_num in range(num_games):
        env = ShitlerEnv()
        game_seed = None if seed is None else seed + game_num
        env.reset(seed=game_seed)

        # Map agents to players based on roles
        agents = {}
        for i, agent_name in enumerate(env.agents):
            role = env.roles[agent_name]
            if role == "lib":
                agents[agent_name] = liberal_agent
            else:
                agents[agent_name] = fascist_agent

        # Play game
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

        # Determine winner and win condition
        for agent_name, reward in env.rewards.items():
            if reward == 1:
                role = env.roles[agent_name]
                if role == "lib":
                    results["lib_wins"] += 1
                    if env.lib_policies >= 5:
                        results["win_conditions"]["lib_5_policies"] += 1
                    else:
                        results["win_conditions"]["hitler_executed"] += 1
                else:
                    results["fasc_wins"] += 1
                    if env.fasc_policies >= 6:
                        results["win_conditions"]["fasc_6_policies"] += 1
                    else:
                        results["win_conditions"]["hitler_chancellor"] += 1
                break

        if verbose and (game_num + 1) % 100 == 0:
            print(f"  Completed {game_num + 1}/{num_games} games")

    results["lib_win_rate"] = results["lib_wins"] / num_games
    results["fasc_win_rate"] = results["fasc_wins"] / num_games
    results["num_games"] = num_games
    results["win_conditions"] = dict(results["win_conditions"])

    return results


def run_experiment(cfr_checkpoint, num_games=1000, seed=42):
    """Run selfish vs CFR experiments."""
    results = {}

    print("=" * 60)
    print("SELFISH vs CFR AGENT EVALUATION")
    print("=" * 60)
    print(f"CFR Checkpoint: {cfr_checkpoint}")
    print(f"Games per experiment: {num_games}")
    print(f"Seed: {seed}")
    print()

    # Load CFR agent
    cfr_agent = CFRAgent(checkpoint_path=cfr_checkpoint)
    selfish_agent = SelfishAgent()
    random_agent = SimpleRandomAgent()

    # Experiment 1: CFR Liberals vs Random Fascists (baseline for CFR)
    print("1. CFR Liberals vs Random Fascists")
    print("-" * 40)
    cfr_vs_random = evaluate_role_based_agents(
        liberal_agent=cfr_agent,
        fascist_agent=random_agent,
        num_games=num_games,
        seed=seed
    )
    results["cfr_lib_vs_random_fasc"] = {
        "lib_win_rate": cfr_vs_random["lib_win_rate"],
        "fasc_win_rate": cfr_vs_random["fasc_win_rate"],
        "win_conditions": cfr_vs_random["win_conditions"],
    }
    print(f"  Liberal win rate: {cfr_vs_random['lib_win_rate']:.2%}")
    print(f"  Win conditions: {cfr_vs_random['win_conditions']}")
    print()

    # Experiment 2: Selfish Liberals vs Random Fascists (baseline for Selfish)
    print("2. Selfish Liberals vs Random Fascists")
    print("-" * 40)
    selfish_vs_random = evaluate_role_based_agents(
        liberal_agent=selfish_agent,
        fascist_agent=random_agent,
        num_games=num_games,
        seed=seed
    )
    results["selfish_lib_vs_random_fasc"] = {
        "lib_win_rate": selfish_vs_random["lib_win_rate"],
        "fasc_win_rate": selfish_vs_random["fasc_win_rate"],
        "win_conditions": selfish_vs_random["win_conditions"],
    }
    print(f"  Liberal win rate: {selfish_vs_random['lib_win_rate']:.2%}")
    print(f"  Win conditions: {selfish_vs_random['win_conditions']}")
    print()

    # Experiment 3: CFR Liberals vs Selfish Fascists
    print("3. CFR Liberals vs Selfish Fascists")
    print("-" * 40)
    cfr_vs_selfish = evaluate_role_based_agents(
        liberal_agent=cfr_agent,
        fascist_agent=selfish_agent,
        num_games=num_games,
        seed=seed
    )
    results["cfr_lib_vs_selfish_fasc"] = {
        "lib_win_rate": cfr_vs_selfish["lib_win_rate"],
        "fasc_win_rate": cfr_vs_selfish["fasc_win_rate"],
        "win_conditions": cfr_vs_selfish["win_conditions"],
    }
    print(f"  Liberal win rate: {cfr_vs_selfish['lib_win_rate']:.2%}")
    print(f"  Win conditions: {cfr_vs_selfish['win_conditions']}")
    print()

    # Experiment 4: Selfish Liberals vs Selfish Fascists
    print("4. Selfish Liberals vs Selfish Fascists")
    print("-" * 40)
    selfish_vs_selfish = evaluate_role_based_agents(
        liberal_agent=selfish_agent,
        fascist_agent=selfish_agent,
        num_games=num_games,
        seed=seed
    )
    results["selfish_lib_vs_selfish_fasc"] = {
        "lib_win_rate": selfish_vs_selfish["lib_win_rate"],
        "fasc_win_rate": selfish_vs_selfish["fasc_win_rate"],
        "win_conditions": selfish_vs_selfish["win_conditions"],
    }
    print(f"  Liberal win rate: {selfish_vs_selfish['lib_win_rate']:.2%}")
    print(f"  Win conditions: {selfish_vs_selfish['win_conditions']}")
    print()

    # Experiment 5: Random Liberals vs Random Fascists (baseline)
    print("5. Random Liberals vs Random Fascists (baseline)")
    print("-" * 40)
    random_vs_random = evaluate_role_based_agents(
        liberal_agent=random_agent,
        fascist_agent=random_agent,
        num_games=num_games,
        seed=seed
    )
    results["random_vs_random"] = {
        "lib_win_rate": random_vs_random["lib_win_rate"],
        "fasc_win_rate": random_vs_random["fasc_win_rate"],
        "win_conditions": random_vs_random["win_conditions"],
    }
    print(f"  Liberal win rate: {random_vs_random['lib_win_rate']:.2%}")
    print(f"  Win conditions: {random_vs_random['win_conditions']}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<40} {'Lib Win %':>10}")
    print("-" * 50)
    print(f"{'Random vs Random (baseline)':<40} {results['random_vs_random']['lib_win_rate']*100:>9.1f}%")
    print(f"{'Selfish Lib vs Random Fasc':<40} {results['selfish_lib_vs_random_fasc']['lib_win_rate']*100:>9.1f}%")
    print(f"{'CFR Lib vs Random Fasc':<40} {results['cfr_lib_vs_random_fasc']['lib_win_rate']*100:>9.1f}%")
    print(f"{'Selfish Lib vs Selfish Fasc':<40} {results['selfish_lib_vs_selfish_fasc']['lib_win_rate']*100:>9.1f}%")
    print(f"{'CFR Lib vs Selfish Fasc':<40} {results['cfr_lib_vs_selfish_fasc']['lib_win_rate']*100:>9.1f}%")
    print("=" * 60)

    # Analysis
    print("\nANALYSIS:")
    cfr_improvement = results['cfr_lib_vs_random_fasc']['lib_win_rate'] - results['random_vs_random']['lib_win_rate']
    selfish_improvement = results['selfish_lib_vs_random_fasc']['lib_win_rate'] - results['random_vs_random']['lib_win_rate']
    print(f"  CFR improvement over random: {cfr_improvement*100:+.1f}%")
    print(f"  Selfish improvement over random: {selfish_improvement*100:+.1f}%")

    if results['cfr_lib_vs_random_fasc']['lib_win_rate'] > results['selfish_lib_vs_random_fasc']['lib_win_rate']:
        diff = results['cfr_lib_vs_random_fasc']['lib_win_rate'] - results['selfish_lib_vs_random_fasc']['lib_win_rate']
        print(f"  CFR beats Selfish by: {diff*100:.1f}%")
    else:
        diff = results['selfish_lib_vs_random_fasc']['lib_win_rate'] - results['cfr_lib_vs_random_fasc']['lib_win_rate']
        print(f"  Selfish beats CFR by: {diff*100:.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Selfish vs CFR Agent")
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/cfr/checkpoints/cfr_liberal_iter_350000.pkl",
                        help="Path to CFR checkpoint")
    parser.add_argument("--games", type=int, default=1000, help="Games per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent.parent.parent / args.checkpoint

    results = run_experiment(
        cfr_checkpoint=str(checkpoint_path),
        num_games=args.games,
        seed=args.seed
    )

    if args.save:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"selfish_vs_cfr_{timestamp}.json"

        results["metadata"] = {
            "cfr_checkpoint": str(checkpoint_path),
            "num_games": args.games,
            "seed": args.seed,
            "timestamp": timestamp,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_file}")
