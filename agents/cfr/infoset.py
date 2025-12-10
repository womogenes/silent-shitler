"""Information set abstraction for CFR.

Uses per-player feature vectors instead of heuristic suspicion scores.
This allows CFR to learn what patterns actually matter.
"""


def get_player_features(obs, player_idx):
    """
    Compute feature vector for a single player based on public history.
    
    Returns tuple of bucketed features:
    - fasc_as_prez: # of fascist policies enacted as president (0, 1, 2+)
    - fasc_as_chanc: # of fascist policies enacted as chancellor (0, 1, 2+)
    - lib_as_prez: # of liberal policies enacted as president (0, 1, 2+)
    - lib_as_chanc: # of liberal policies enacted as chancellor (0, 1, 2+)
    - claim_conflicts: # of claim conflicts involved in (0, 1+)
    """
    fasc_as_prez = 0
    fasc_as_chanc = 0
    lib_as_prez = 0
    lib_as_chanc = 0
    claim_conflicts = 0
    
    hist_len = len(obs["hist_president"])
    for i in range(hist_len):
        succeeded = obs["hist_succeeded"][i]
        if succeeded != 1:
            continue
        
        prez = obs["hist_president"][i]
        chanc = obs["hist_chancellor"][i]
        policy = obs["hist_policy"][i]
        prez_claim = obs["hist_prez_claim"][i]
        chanc_claim = obs["hist_chanc_claim"][i]
        
        # Count policies by role
        if policy == 1:  # fascist
            if prez == player_idx:
                fasc_as_prez += 1
            if chanc == player_idx:
                fasc_as_chanc += 1
        else:  # liberal
            if prez == player_idx:
                lib_as_prez += 1
            if chanc == player_idx:
                lib_as_chanc += 1
        
        # Check for claim conflicts
        if prez_claim >= 0 and chanc_claim >= 0:
            # Prez claims X fascist in 3 cards, discards 1, chanc sees 2
            # Valid: chanc_claim in [max(0, prez_claim-1), min(2, prez_claim)]
            expected_min = max(0, prez_claim - 1)
            expected_max = min(2, prez_claim)
            if chanc_claim < expected_min or chanc_claim > expected_max:
                if prez == player_idx or chanc == player_idx:
                    claim_conflicts += 1
    
    # Bucket the features
    return (
        min(fasc_as_prez, 2),
        min(fasc_as_chanc, 2),
        min(lib_as_prez, 2),
        min(lib_as_chanc, 2),
        min(claim_conflicts, 1),
    )


def get_history_abstraction(obs):
    """
    Get per-player feature vectors for all 5 players.
    
    Returns tuple of 5 feature tuples.
    """
    return tuple(get_player_features(obs, i) for i in range(5))


# Legacy functions for backward compatibility
def compute_suspicion_scores(obs):
    """Legacy function - computes simple suspicion scores."""
    scores = [0.0] * 5
    hist_len = len(obs["hist_president"])
    for i in range(hist_len):
        succeeded = obs["hist_succeeded"][i]
        if succeeded != 1:
            continue
        prez = obs["hist_president"][i]
        chanc = obs["hist_chancellor"][i]
        policy = obs["hist_policy"][i]
        if policy == 1:
            scores[prez] += 1.0
            scores[chanc] += 1.0
    return scores


def bucket_suspicion(scores):
    """Legacy function - buckets suspicion scores."""
    buckets = []
    max_score = max(scores) if max(scores) > 0 else 1.0
    for s in scores:
        normalized = s / max_score
        bucket = min(int(normalized * 2), 2)
        buckets.append(bucket)
    return tuple(buckets)


def get_infoset_key(obs, phase, agent_idx):
    """
    Generate a hashable information set key from observation.
    
    Fascists include all_roles; liberals use per-player feature vectors.
    """
    role = obs["role"]
    is_fascist = role in [1, 2]  # fasc or hitty
    
    # Core game state
    key = [
        role,
        obs["lib_policies"],
        obs["fasc_policies"],
        obs["election_tracker"],
        phase,
        tuple(obs["executed"]),
        obs["president_idx"],
        obs["chancellor_nominee"],
    ]
    
    # Role knowledge / history abstraction
    if is_fascist:
        key.append(tuple(obs["all_roles"]))
        key.append(None)  # no history abstraction needed
    else:
        key.append(None)  # no role knowledge
        key.append(get_history_abstraction(obs))  # per-player features
    
    # Phase-specific info
    if phase in ["prez_cardsel", "chanc_cardsel"]:
        if "cards" in obs:
            key.append(tuple(obs["cards"]))
        else:
            key.append(None)
    else:
        key.append(None)
    
    return tuple(key)
