"""Information set abstraction and bucketing for CFR."""

# Configurable suspicion bucket count
NUM_SUSPICION_BUCKETS = 5


def compute_suspicion_scores(obs):
    """
    Compute raw suspicion scores for each player based on public history.
    
    Features considered:
    - claim_conflicts: prez/chanc claims don't add up
    - fasc_participation: was in government that played fascist
    - fasc_votes: voted yes on governments that played fascist
    """
    scores = [0.0] * 5
    
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
        votes = obs["hist_votes"][i]
        
        # Fascist policy participation
        if policy == 1:  # fascist policy
            scores[prez] += 1.0
            scores[chanc] += 1.0
            # Voted yes on fasc government
            for p in range(5):
                if votes[p] == 1:
                    scores[p] += 0.3
        
        # Claim conflicts: prez sees 3, discards 1, chanc sees 2
        # So chanc_claim should be prez_claim or prez_claim - 1
        if prez_claim >= 0 and chanc_claim >= 0:
            expected_chanc_min = max(0, prez_claim - 1)
            expected_chanc_max = min(2, prez_claim)
            if chanc_claim < expected_chanc_min or chanc_claim > expected_chanc_max:
                # Conflict - both are suspicious
                scores[prez] += 0.5
                scores[chanc] += 0.5
    
    # Execution choices (executing libs is suspicious for fascists)
    for i in range(hist_len):
        exec_target = obs["hist_execution"][i]
        if exec_target >= 0:
            prez = obs["hist_president"][i]
            # Can't determine if target was lib without role info
            # Just note that executions happened
            # TODO: track execution patterns
    
    return scores


def bucket_suspicion(scores):
    """Convert raw suspicion scores to bucket indices (0 to NUM_SUSPICION_BUCKETS-1)."""
    buckets = []
    max_score = max(scores) if max(scores) > 0 else 1.0
    for s in scores:
        normalized = s / max_score
        bucket = int(normalized * (NUM_SUSPICION_BUCKETS - 1))
        bucket = min(bucket, NUM_SUSPICION_BUCKETS - 1)
        buckets.append(bucket)
    return tuple(buckets)


def get_infoset_key(obs, phase, agent_idx):
    """
    Generate a hashable information set key from observation.
    
    Fascists include all_roles; liberals use suspicion buckets.
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
    
    # Role knowledge
    if is_fascist:
        key.append(tuple(obs["all_roles"]))
        key.append(None)  # no suspicion needed
    else:
        key.append(None)  # no role knowledge
        scores = compute_suspicion_scores(obs)
        key.append(bucket_suspicion(scores))
    
    # Phase-specific info
    if phase in ["prez_cardsel", "chanc_cardsel"]:
        if "cards" in obs:
            key.append(tuple(obs["cards"]))
        else:
            key.append(None)
    else:
        key.append(None)
    
    return tuple(key)
