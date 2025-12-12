"""Game state management for DeepRole CFR solving."""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv


def create_game_at_state(lib_policies, fasc_policies, president_idx=0, seed=None):
    """Create a game environment at a specific state.

    Args:
        lib_policies: Number of liberal policies enacted
        fasc_policies: Number of fascist policies enacted
        president_idx: Current president (0-4)
        seed: Random seed for role assignment

    Returns:
        Configured ShitlerEnv at specified state
    """
    env = ShitlerEnv()
    env.reset(seed=seed)

    # Set policy counts
    env.lib_policies = lib_policies
    env.fasc_policies = fasc_policies

    # Set president
    env.president_idx = president_idx

    # Initialize required fields for observe to work
    env.chancellor_nominee = None
    env.prez_cards = []
    env.chanc_cards = []
    env.executed = set()
    env.hist_chancellor = []
    env.hist_president = []
    env.hist_rejected_chancellors = []
    env.hist_policy = []

    # Ensure we're in the right phase
    env.phase = "nomination"
    env.agent_selection = env.agents[president_idx]

    # Set deck state (approximate - would need exact history for perfect restoration)
    # Cards used: at least lib_policies + fasc_policies played, plus discards
    cards_used = lib_policies + fasc_policies
    # Estimate 2 discards per government (president + chancellor)
    estimated_discards = cards_used * 2

    # Reset deck with appropriate cards removed
    total_lib_in_deck = max(0, 6 - lib_policies)
    total_fasc_in_deck = max(0, 11 - fasc_policies)

    # Account for cards in discard
    if estimated_discards > 0:
        # Distribute discards proportionally
        lib_discarded = min(estimated_discards // 3, total_lib_in_deck)
        fasc_discarded = estimated_discards - lib_discarded

        env.deck = [0] * (total_lib_in_deck - lib_discarded) + [1] * (total_fasc_in_deck - fasc_discarded)
        env.discard = [0] * lib_discarded + [1] * fasc_discarded
    else:
        env.deck = [0] * total_lib_in_deck + [1] * total_fasc_in_deck
        env.discard = []

    # Shuffle deck
    np.random.shuffle(env.deck)

    # Create minimal history for deductions
    env.hist_president = []
    env.hist_chancellor = []
    env.hist_votes = []
    env.hist_succeeded = []
    env.hist_policy = []
    env.hist_prez_claim = []
    env.hist_chanc_claim = []
    env.hist_execution = []

    # Add dummy history entries for enacted policies
    for i in range(lib_policies + fasc_policies):
        policy = 0 if i < lib_policies else 1
        # Random president/chancellor for past governments
        prez = np.random.randint(5)
        chanc = np.random.randint(5)
        while chanc == prez:
            chanc = np.random.randint(5)

        env.hist_president.append(prez)
        env.hist_chancellor.append(chanc)
        env.hist_votes.append([1, 1, 1, 0, 0])  # Dummy votes
        env.hist_succeeded.append(1)
        env.hist_policy.append(policy)
        env.hist_prez_claim.append(-1)  # Unknown claims
        env.hist_chanc_claim.append(-1)
        env.hist_execution.append(-1)

    # Check for terminal conditions
    if lib_policies >= 5 or fasc_policies >= 6:
        env._check_game_end()

    return env


def is_terminal_state(lib_policies, fasc_policies):
    """Check if game state is terminal."""
    return lib_policies >= 5 or fasc_policies >= 6


def get_valid_game_states():
    """Get all valid (non-terminal) game states for training.

    Returns list of (lib_policies, fasc_policies) tuples.
    """
    states = []
    for lib in range(5):  # 0-4 liberal policies
        for fasc in range(6):  # 0-5 fascist policies
            if not is_terminal_state(lib, fasc):
                states.append((lib, fasc))
    return states


def get_terminal_states():
    """Get all terminal game states.

    Returns list of (lib_policies, fasc_policies) tuples.
    """
    states = []
    # Liberal wins
    for fasc in range(6):
        states.append((5, fasc))
    # Fascist wins
    for lib in range(5):
        states.append((lib, 6))
    return states


def get_state_dependencies():
    """Get dependency graph for backwards training.

    Returns dict mapping (lib, fasc) -> list of successor states.
    """
    deps = {}
    for lib, fasc in get_valid_game_states():
        successors = []
        # Can reach (lib+1, fasc) or (lib, fasc+1)
        if lib + 1 <= 5:
            successors.append((lib + 1, fasc))
        if fasc + 1 <= 6:
            successors.append((lib, fasc + 1))
        deps[(lib, fasc)] = successors
    return deps