"""Cross-play analysis: CFR vs Random agents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pickle
import random
from collections import defaultdict

from shitler_env.game import ShitlerEnv
from agents.cfr.cfr_agent import CFRAgent, get_legal_actions
from agents.cfr.infoset import get_infoset_key

ROLE_MAP = {"lib": 0, "fasc": 1, "hitty": 2}
ROLE_NAMES = {0: "Liberal", 1: "Fascist", 2: "Hitler"}


def get_roles_list(env):
    return [ROLE_MAP[env.roles[f"P{i}"]] for i in range(5)]


def load_cfr_agent(checkpoint_path):
    """Load CFR agent from checkpoint."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    regret_sums = defaultdict(lambda: defaultdict(float))
    strategy_sums = defaultdict(lambda: defaultdict(float))
    for k, v in data["regret_sums"].items():
        regret_sums[k] = v
    for k, v in data["strategy_sums"].items():
        strategy_sums[k] = v

    cfr_agent = CFRAgent()
    cfr_agent.regret_sums = regret_sums
    cfr_agent.strategy_sums = strategy_sums
    return cfr_agent, len(data["regret_sums"])


def get_random_action(legal_actions):
    return random.choice(legal_actions)


def get_cfr_action(cfr_agent, obs, phase, player_idx, legal_actions):
    infoset_key = get_infoset_key(obs, phase, player_idx)
    strategy = cfr_agent.get_average_strategy(infoset_key, legal_actions)
    return cfr_agent.sample_action(strategy), strategy, infoset_key


def run_games(cfr_agent, num_games, lib_policy, fasc_policy):
    """Run games with specified policies for each team."""
    results = {
        "liberal_wins": 0,
        "fascist_wins": 0,
        "win_conditions": defaultdict(int)
    }

    for _ in range(num_games):
        env = ShitlerEnv()
        env.reset()
        roles = get_roles_list(env)
        lib_players = {i for i, r in enumerate(roles) if r == 0}

        while not all(env.terminations.values()):
            current_agent = env.agent_selection
            current_idx = env.agents.index(current_agent)
            obs = env.observe(current_agent)
            phase = env.phase

            legal_actions = get_legal_actions(env, current_agent)
            if not legal_actions:
                env.step(0)
                continue

            if current_idx in lib_players:
                if lib_policy == "cfr":
                    action, _, _ = get_cfr_action(cfr_agent, obs, phase, current_idx, legal_actions)
                else:
                    action = get_random_action(legal_actions)
            else:
                if fasc_policy == "cfr":
                    action, _, _ = get_cfr_action(cfr_agent, obs, phase, current_idx, legal_actions)
                else:
                    action = get_random_action(legal_actions)

            env.step(action)

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


def run_verbose_game(cfr_agent, lib_policy, fasc_policy, seed=None):
    """Run a single game with verbose output."""
    if seed is not None:
        random.seed(seed)

    env = ShitlerEnv()
    env.reset()
    roles = get_roles_list(env)
    lib_players = {i for i, r in enumerate(roles) if r == 0}
    hitler_idx = roles.index(2)

    print("=" * 70)
    print(f"GAME: {lib_policy.upper()} Liberals vs {fasc_policy.upper()} Fascists")
    print("=" * 70)
    print("Roles:")
    for i, role in enumerate(roles):
        print(f"  P{i}: {ROLE_NAMES[role]}")
    print()

    turn = 0
    while not all(env.terminations.values()):
        current_agent = env.agent_selection
        current_idx = env.agents.index(current_agent)
        obs = env.observe(current_agent)
        phase = env.phase

        legal_actions = get_legal_actions(env, current_agent)
        if not legal_actions:
            env.step(0)
            continue

        role = roles[current_idx]
        role_name = ROLE_NAMES[role]
        is_lib = current_idx in lib_players
        policy = lib_policy if is_lib else fasc_policy

        # Get action
        if policy == "cfr":
            action, strategy, infoset_key = get_cfr_action(
                cfr_agent, obs, phase, current_idx, legal_actions
            )
            # Check if infoset was seen during training
            infoset_seen = infoset_key in cfr_agent.strategy_sums
        else:
            action = get_random_action(legal_actions)
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}
            infoset_seen = None

        # Print decision
        print(f"[Turn {turn}] P{current_idx} ({role_name}) - {phase}")
        print(f"  Policy: {policy.upper()}")
        if infoset_seen is not None:
            print(f"  Infoset seen in training: {infoset_seen}")

        # Phase-specific output
        if phase == "nomination":
            print(f"  Legal nominees: {legal_actions}")
            print(f"  Strategy: {dict(sorted(strategy.items()))}")
            print(f"  -> Nominates P{action} as Chancellor")

        elif phase == "voting":
            ja_prob = strategy.get(1, 0)
            nein_prob = strategy.get(0, 0)
            vote = "JA" if action == 1 else "NEIN"
            nominee = env.chancellor_nominee
            nominee_role = ROLE_NAMES[roles[nominee]]
            is_hitler = nominee == hitler_idx
            print(f"  Voting on: P{nominee} ({nominee_role}){' [HITLER!]' if is_hitler else ''}")
            print(f"  Strategy: JA={ja_prob:.1%}, NEIN={nein_prob:.1%}")
            print(f"  -> Votes {vote}")

        elif phase in ["prez_cardsel", "chanc_cardsel"]:
            cards_onehot = obs.get("cards", [])
            # Decode one-hot: index = num_libs
            try:
                num_libs = cards_onehot.index(1)
            except ValueError:
                num_libs = 0
            if phase == "prez_cardsel":
                num_fasc = 3 - num_libs
            else:
                num_fasc = 2 - num_libs
            print(f"  Hand: {num_libs} Liberal, {num_fasc} Fascist")
            discard_lib_prob = strategy.get(0, 0)
            discard_fasc_prob = strategy.get(1, 0)
            print(f"  Strategy: discard_lib={discard_lib_prob:.1%}, discard_fasc={discard_fasc_prob:.1%}")
            discarded = "Liberal" if action == 0 else "Fascist"
            print(f"  -> Discards {discarded}")

        elif phase in ["prez_claim", "chanc_claim"]:
            print(f"  Strategy: {dict(sorted(strategy.items()))}")
            print(f"  -> Claims {action} fascist cards")

        elif phase == "execution":
            print(f"  Legal targets: {legal_actions}")
            print(f"  Strategy: {dict(sorted(strategy.items()))}")
            target_role = ROLE_NAMES[roles[action]]
            print(f"  -> Executes P{action} ({target_role})")

        print()
        env.step(action)
        turn += 1

        # Print state after certain phases
        if phase == "chanc_claim":
            print(f"  >> State: {env.lib_policies} Lib, {env.fasc_policies} Fasc policies")
            print()

    # Print result
    print("=" * 70)
    print("GAME OVER")
    print("=" * 70)
    lib_idx = roles.index(0)
    lib_reward = env.rewards[f"P{lib_idx}"]

    if lib_reward > 0:
        if env.lib_policies >= 5:
            print("LIBERALS WIN - 5 Liberal Policies!")
        else:
            print("LIBERALS WIN - Hitler Executed!")
    else:
        if env.fasc_policies >= 6:
            print("FASCISTS WIN - 6 Fascist Policies!")
        else:
            print("FASCISTS WIN - Hitler Elected Chancellor!")

    print(f"Final: {env.lib_policies} Lib, {env.fasc_policies} Fasc policies")


def main():
    checkpoint_path = Path(__file__).parent / "checkpoints" / "cfr_iter_500000.pkl"
    cfr_agent, num_infosets = load_cfr_agent(checkpoint_path)
    print(f"Loaded {num_infosets:,} infosets")

    NUM_GAMES = 500

    print()
    print("=" * 70)
    print("CROSS-PLAY ANALYSIS")
    print("=" * 70)

    # Random vs Random
    print("\nRunning Random vs Random...")
    rr = run_games(cfr_agent, NUM_GAMES, "random", "random")
    print(f"  Liberal Win Rate: {rr['liberal_wins']/NUM_GAMES*100:.1f}%")

    # CFR Liberals vs Random Fascists
    print("\nRunning CFR Liberals vs Random Fascists...")
    cr = run_games(cfr_agent, NUM_GAMES, "cfr", "random")
    print(f"  Liberal Win Rate: {cr['liberal_wins']/NUM_GAMES*100:.1f}%")

    # Random Liberals vs CFR Fascists
    print("\nRunning Random Liberals vs CFR Fascists...")
    rc = run_games(cfr_agent, NUM_GAMES, "random", "cfr")
    print(f"  Liberal Win Rate: {rc['liberal_wins']/NUM_GAMES*100:.1f}%")

    # CFR vs CFR
    print("\nRunning CFR vs CFR...")
    cc = run_games(cfr_agent, NUM_GAMES, "cfr", "cfr")
    print(f"  Liberal Win Rate: {cc['liberal_wins']/NUM_GAMES*100:.1f}%")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Matchup':<40} {'Liberal Win %':>15}")
    print("-" * 55)
    print(f"{'Random vs Random (baseline)':<40} {rr['liberal_wins']/NUM_GAMES*100:>14.1f}%")
    print(f"{'CFR Liberals vs Random Fascists':<40} {cr['liberal_wins']/NUM_GAMES*100:>14.1f}%")
    print(f"{'Random Liberals vs CFR Fascists':<40} {rc['liberal_wins']/NUM_GAMES*100:>14.1f}%")
    print(f"{'CFR vs CFR (self-play)':<40} {cc['liberal_wins']/NUM_GAMES*100:>14.1f}%")

    # Run verbose example game
    print("\n" + "=" * 70)
    print("EXAMPLE GAME: CFR Liberals vs Random Fascists")
    print("=" * 70 + "\n")
    run_verbose_game(cfr_agent, "cfr", "random", seed=42)


if __name__ == "__main__":
    main()
