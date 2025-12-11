"""
Simplified information set abstraction for CFR with coarser bucketing.
Uses only 3 suspicion levels per player to dramatically reduce state space.
"""


def get_player_suspicion(obs, player_idx):
    """
    Compute simple suspicion score for a single player.

    Returns: 0 (clean), 1 (suspicious), 2 (dirty)
    """
    fascist_count = 0

    hist_len = len(obs["hist_president"])
    for i in range(hist_len):
        succeeded = obs["hist_succeeded"][i]
        if succeeded != 1:
            continue

        prez = obs["hist_president"][i]
        chanc = obs["hist_chancellor"][i]
        policy = obs["hist_policy"][i]

        if policy == 1:  # fascist policy
            if prez == player_idx:
                fascist_count += 1
            if chanc == player_idx:
                fascist_count += 1

    # Simple 3-bucket suspicion
    if fascist_count == 0:
        return 0  # clean
    elif fascist_count <= 2:
        return 1  # suspicious
    else:
        return 2  # dirty


def get_simple_history_abstraction(obs):
    """
    Get simplified suspicion levels for all 5 players.

    Returns tuple of 5 suspicion levels (0-2 each).
    """
    return tuple(get_player_suspicion(obs, i) for i in range(5))


def get_infoset_key_simple(obs, phase, agent_idx):
    """
    Generate a simplified hashable information set key.

    Much coarser abstraction to reduce state space:
    - Only 3 suspicion levels per player
    - Less granular policy tracking
    """
    role = obs["role"]
    is_fascist = role in [1, 2]  # fasc or hitty

    # Bucket policies into ranges for coarser abstraction
    lib_policies = min(obs["lib_policies"], 4)  # 0-4+
    fasc_policies = min(obs["fasc_policies"], 5)  # 0-5+

    # Core game state (coarsened)
    key = [
        role,
        lib_policies,
        fasc_policies,
        min(obs["election_tracker"], 2),  # 0, 1, 2+
        phase,
        any(obs["executed"]),  # boolean: has anyone been executed
        obs["president_idx"],
    ]

    # Simplified chancellor tracking
    if obs["chancellor_nominee"] >= 0:
        # Only track if chancellor was previously president
        chanc_was_prez = obs["chancellor_nominee"] in [
            obs.get("hist_president", [])[i]
            for i in range(len(obs.get("hist_president", [])))
        ]
        key.append(chanc_was_prez)
    else:
        key.append(None)

    # Role knowledge / history abstraction
    if is_fascist:
        key.append(tuple(obs["all_roles"]))
        key.append(None)  # no history abstraction needed
    else:
        key.append(None)  # no role knowledge
        key.append(get_simple_history_abstraction(obs))  # 3^5 = 243 combinations

    # Phase-specific info
    if phase in ["prez_cardsel", "chanc_cardsel"]:
        if "cards" in obs:
            # Just track number of each type, not order
            num_libs = sum(1 for c in obs["cards"] if c == 0)
            num_fascs = len(obs["cards"]) - num_libs
            key.append((num_libs, num_fascs))
        else:
            key.append(None)
    else:
        key.append(None)

    return tuple(key)


def calculate_state_space_size():
    """
    Calculate theoretical state space size with simplified abstraction.
    """
    # Liberal perspective
    roles = 1  # Liberal only sees own role
    lib_policies = 5  # 0-4+
    fasc_policies = 6  # 0-5+
    election_tracker = 3  # 0, 1, 2+
    phases = 7  # nomination, voting, prez_cardsel, chanc_cardsel, prez_claim, chanc_claim, execution
    executed = 2  # boolean
    president_idx = 5  # 0-4
    chanc_was_prez = 2  # boolean (simplified)
    suspicion_combos = 3**5  # 3 levels per player, 5 players = 243

    liberal_states = (
        roles * lib_policies * fasc_policies * election_tracker *
        phases * executed * president_idx * chanc_was_prez * suspicion_combos
    )

    # Fascist perspective (knows all roles, no history abstraction needed)
    role_configs = 2  # fascist or hitler
    all_roles = 20  # C(5,3) ways to pick 3 liberals from 5 players

    fascist_states = (
        role_configs * lib_policies * fasc_policies * election_tracker *
        phases * executed * president_idx * chanc_was_prez * all_roles
    )

    total_states = liberal_states + fascist_states

    return {
        "liberal_states": liberal_states,
        "fascist_states": fascist_states,
        "total_states": total_states,
        "suspicion_combinations": suspicion_combos,
    }


# For backward compatibility, keep the original function name
get_infoset_key = get_infoset_key_simple


if __name__ == "__main__":
    # Print state space analysis
    sizes = calculate_state_space_size()

    print("=" * 70)
    print("SIMPLIFIED CFR STATE SPACE ANALYSIS")
    print("=" * 70)
    print(f"Suspicion levels per player: 3 (clean/suspicious/dirty)")
    print(f"Suspicion combinations: 3^5 = {sizes['suspicion_combinations']:,}")
    print()
    print(f"Liberal perspective states: {sizes['liberal_states']:,}")
    print(f"Fascist perspective states: {sizes['fascist_states']:,}")
    print(f"TOTAL THEORETICAL STATES: {sizes['total_states']:,}")
    print()
    print("Comparison to original:")
    print("  Original: ~11,600,000,000 states (11.6 billion)")
    print(f"  Simplified: {sizes['total_states']:,} states")
    print(f"  Reduction: {11600000000 / sizes['total_states']:.0f}x smaller")
    print("=" * 70)