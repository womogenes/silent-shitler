"""Debug CFR gameplay by recording detailed transcripts."""

import sys
from pathlib import Path
import pickle
import random
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from agents.cfr.cfr_agent import CFRAgent
from agents.cfr.infoset import get_infoset_key, get_player_features

ROLE_MAP = {"lib": 0, "fasc": 1, "hitty": 2}
ROLE_NAMES = {0: "Liberal", 1: "Fascist", 2: "Hitler"}


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
    return cfr_agent, len(data["strategy_sums"])


def get_legal_actions_from_obs(obs, action_space):
    """Extract legal actions from observation masks."""
    if "nomination_mask" in obs:
        return [i for i, v in enumerate(obs["nomination_mask"]) if v == 1]
    elif "execution_mask" in obs:
        return [i for i, v in enumerate(obs["execution_mask"]) if v == 1]
    elif "card_action_mask" in obs:
        return [i for i, v in enumerate(obs["card_action_mask"]) if v == 1]
    else:
        return list(range(action_space.n))


def format_infoset_key(key, obs):
    """Format infoset key for readable output."""
    role, lib_pol, fasc_pol, elec_track, phase, executed, prez_idx, chanc_nom, roles_or_none, hist_or_none, cards = key

    formatted = {
        "role": ROLE_NAMES[role],
        "lib_policies": lib_pol,
        "fasc_policies": fasc_pol,
        "election_tracker": elec_track,
        "phase": phase,
        "executed": [i for i, e in enumerate(executed) if e == 1],
        "president_idx": prez_idx,
        "chancellor_nominee": chanc_nom if chanc_nom != -1 else None,
    }

    # Add role knowledge (fascists only)
    if roles_or_none is not None:
        formatted["all_roles"] = [ROLE_NAMES[r] for r in roles_or_none]

    # Add history abstraction (liberals only)
    if hist_or_none is not None:
        formatted["player_features"] = {}
        for i, features in enumerate(hist_or_none):
            fasc_prez, fasc_chanc, lib_prez, lib_chanc, conflicts = features
            formatted["player_features"][f"P{i}"] = {
                "fasc_as_prez": fasc_prez,
                "fasc_as_chanc": fasc_chanc,
                "lib_as_prez": lib_prez,
                "lib_as_chanc": lib_chanc,
                "claim_conflicts": conflicts,
            }

    # Add cards if present
    if cards is not None:
        num_libs = cards.index(1) if 1 in cards else 0
        num_fascs = len(cards) - 1 - num_libs
        formatted["cards"] = f"{num_libs}L, {num_fascs}F"

    return formatted


def play_game_with_transcript(cfr_agent, num_infosets, liberal_uses_cfr=True, seed=None, log_file=None):
    """Play a single game and record detailed transcript."""
    if seed is not None:
        random.seed(seed)

    env = ShitlerEnv()
    env.reset(seed=seed)

    # Get roles
    roles = {agent: ROLE_MAP[env.roles[agent]] for agent in env.agents}
    player_indices = {agent: i for i, agent in enumerate(env.agents)}

    def log(msg):
        """Print and optionally write to log file."""
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    log("=" * 80)
    log(f"GAME TRANSCRIPT (seed={seed})")
    log("=" * 80)
    log("Roles:")
    for agent, role in roles.items():
        idx = player_indices[agent]
        log(f"  P{idx} ({agent}): {ROLE_NAMES[role]}")
    log("")

    transcript_summary = {
        "seed": seed,
        "roles": {f"P{i}": ROLE_NAMES[roles[agent]] for agent, i in player_indices.items()},
    }

    move_num = 0
    infosets_found = 0
    infosets_not_found = 0

    while not all(env.terminations.values()):
        current_agent = env.agent_selection
        current_idx = player_indices[current_agent]
        current_role = roles[current_agent]
        obs = env.observe(current_agent)
        action_space = env.action_space(current_agent)
        phase = env.phase

        # Get legal actions
        legal_actions = get_legal_actions_from_obs(obs, action_space)
        if not legal_actions:
            env.step(0)
            continue

        # Determine if this player uses CFR
        is_liberal = (current_role == 0)
        uses_cfr = (liberal_uses_cfr and is_liberal) or (not liberal_uses_cfr and not is_liberal)

        move_data = {
            "move_num": move_num,
            "player": f"P{current_idx}",
            "role": ROLE_NAMES[current_role],
            "phase": phase,
            "uses_cfr": uses_cfr,
        }

        if uses_cfr:
            # HARD-CODE: Liberals always discard fascist if possible (canonical strategy)
            # This matches the training evaluation and is strategically correct
            if is_liberal and phase in ["prez_cardsel", "chanc_cardsel"] and 1 in legal_actions:
                action = 1  # always discard fascist (play liberal)
                infoset_found = True  # Mark as "found" (hard-coded)
                strategy = {a: (1.0 if a == 1 else 0.0) for a in legal_actions}
                move_data["hard_coded"] = True
                infoset_key = None  # Not used for hard-coded decisions
            else:
                # Get infoset key - CORRECT player index
                infoset_key = get_infoset_key(obs, phase, current_idx)

                # Check if infoset exists in training data
                infoset_found = infoset_key in cfr_agent.strategy_sums

                if infoset_found:
                    infosets_found += 1
                    strategy = cfr_agent.get_average_strategy(infoset_key, legal_actions)
                else:
                    infosets_not_found += 1
                    # Uniform random if not found
                    strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}

                # Sample action
                action = cfr_agent.sample_action(strategy)
                move_data["hard_coded"] = False

            # Record details
            move_data["infoset_found"] = infoset_found
            if infoset_key is not None:
                move_data["infoset_key"] = format_infoset_key(infoset_key, obs)
            move_data["legal_actions"] = legal_actions
            move_data["strategy"] = {a: f"{strategy[a]:.3f}" for a in legal_actions}
            move_data["action"] = action

            # Print verbose output
            log(f"[Move {move_num}] P{current_idx} ({ROLE_NAMES[current_role]}) - {phase}")
            log(f"  Uses CFR: Yes")

            if move_data.get("hard_coded"):
                log(f"  HARD-CODED: Always play liberal policy (canonical strategy)")
                log(f"  Legal actions: {legal_actions}")
                log(f"  Action chosen: {action}")
            else:
                log(f"  Infoset found in training: {'YES' if infoset_found else 'NO (using uniform random)'}")
                if infoset_found:
                    log(f"  Legal actions: {legal_actions}")
                    log(f"  Strategy: {strategy}")
                    log(f"  Action chosen: {action}")

                    # Log infoset details if found
                    if log_file and infoset_key is not None:
                        formatted_infoset = format_infoset_key(infoset_key, obs)
                        log(f"  Infoset details:")
                        for key, value in formatted_infoset.items():
                            if key == "player_features":
                                log(f"    {key}:")
                                for player, features in value.items():
                                    log(f"      {player}: {features}")
                            else:
                                log(f"    {key}: {value}")
                else:
                    log(f"  Legal actions: {legal_actions}")
                    log(f"  Using uniform: {strategy}")
                    log(f"  Action chosen: {action}")
            log("")

        else:
            # Random agent
            action = random.choice(legal_actions)
            move_data["uses_cfr"] = False
            move_data["action"] = action

            log(f"[Move {move_num}] P{current_idx} ({ROLE_NAMES[current_role]}) - {phase}")
            log(f"  Uses CFR: No (random)")
            log(f"  Action: {action}")
            log("")

        env.step(action)
        move_num += 1

        # Print game state after certain phases
        if phase == "chanc_claim" or phase == "execution":
            log(f"  >> State: {env.lib_policies}L, {env.fasc_policies}F")
            log("")

    # Game result
    liberal_reward = env.rewards[env.agents[0]] if roles[env.agents[0]] == 0 else env.rewards[[a for a, r in roles.items() if r == 0][0]]
    winner = "Liberals" if liberal_reward > 0 else "Fascists"

    log("=" * 80)
    log(f"GAME OVER: {winner} win!")
    log(f"Final: {env.lib_policies}L, {env.fasc_policies}F")
    log(f"Infosets found: {infosets_found}/{infosets_found + infosets_not_found} ({100*infosets_found/(infosets_found + infosets_not_found):.1f}%)")
    log("=" * 80)
    log("")

    transcript_summary["result"] = {
        "winner": winner,
        "lib_policies": env.lib_policies,
        "fasc_policies": env.fasc_policies,
        "infosets_found": infosets_found,
        "infosets_not_found": infosets_not_found,
    }

    return transcript_summary


def main():
    print("Loading CFR-Liberal agent...")
    cfr_path = Path(__file__).parent / "checkpoints" / "cfr_liberal_iter_50000.pkl"

    if not cfr_path.exists():
        print(f"ERROR: CFR checkpoint not found at {cfr_path}")
        print("Available checkpoints:")
        for f in (Path(__file__).parent / "checkpoints").glob("*.pkl"):
            print(f"  {f.name}")
        return

    cfr_agent, num_infosets = load_cfr_agent(cfr_path)
    print(f"Loaded {num_infosets:,} infosets")
    print()

    # Open log file
    output_file = Path(__file__).parent / "cfr_debug_transcripts.txt"

    with open(output_file, "w") as log_file:
        log_file.write(f"CFR DEBUG TRANSCRIPTS\n")
        log_file.write(f"Checkpoint: {cfr_path.name}\n")
        log_file.write(f"Total infosets: {num_infosets:,}\n")
        log_file.write(f"\n{'='*80}\n\n")

        # Play 10 games
        all_transcripts = []

        for i in range(10):
            header = f"GAME {i+1}/10"
            print(f"\n{'='*80}")
            print(header)
            print(f"{'='*80}\n")
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"{header}\n")
            log_file.write(f"{'='*80}\n\n")

            transcript = play_game_with_transcript(
                cfr_agent,
                num_infosets,
                liberal_uses_cfr=True,
                seed=42 + i,
                log_file=log_file
            )
            all_transcripts.append(transcript)

    print(f"\nTranscripts saved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    liberal_wins = sum(1 for t in all_transcripts if t["result"]["winner"] == "Liberals")
    total_found = sum(t["result"]["infosets_found"] for t in all_transcripts)
    total_not_found = sum(t["result"]["infosets_not_found"] for t in all_transcripts)

    print(f"Liberal win rate: {liberal_wins}/10 ({100*liberal_wins/10:.0f}%)")
    print(f"Infosets found: {total_found}/{total_found + total_not_found} ({100*total_found/(total_found + total_not_found):.1f}%)")

    print("\nPer-game results:")
    for i, t in enumerate(all_transcripts):
        winner = t["result"]["winner"]
        found = t["result"]["infosets_found"]
        not_found = t["result"]["infosets_not_found"]
        total = found + not_found
        print(f"  Game {i+1}: {winner:10s} - {found}/{total} infosets found ({100*found/total:.1f}%)")

    # Append summary to log file
    with open(output_file, "a") as log_file:
        log_file.write("\n\n" + "=" * 80 + "\n")
        log_file.write("SUMMARY STATISTICS\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Liberal win rate: {liberal_wins}/10 ({100*liberal_wins/10:.0f}%)\n")
        log_file.write(f"Infosets found: {total_found}/{total_found + total_not_found} ({100*total_found/(total_found + total_not_found):.1f}%)\n\n")
        log_file.write("Per-game results:\n")
        for i, t in enumerate(all_transcripts):
            winner = t["result"]["winner"]
            found = t["result"]["infosets_found"]
            not_found = t["result"]["infosets_not_found"]
            total = found + not_found
            log_file.write(f"  Game {i+1}: {winner:10s} - {found}/{total} infosets found ({100*found/total:.1f}%)\n")


if __name__ == "__main__":
    main()
