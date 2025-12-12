"""Belief update implementation for DeepRole.

According to the paper, belief updates require strategies computed during CFR:
b(ρ|h) ∝ b(ρ)(1 - 1{h ⊢ ¬ρ}) ∏_{i∈1...p} π^σ_i(I_i(h, ρ))

The challenge: During actual play, we don't have access to opponent strategies.
This implementation does what we CAN do: logical deduction.
"""

import numpy as np
from typing import Dict, List, Tuple


class BeliefUpdater:
    """Updates beliefs based on observations and logical deduction.

    NOTE: Full DeepRole belief updates require strategies from CFR which
    we don't have during play. This implements the logical consistency
    part (1 - 1{h ⊢ ¬ρ}) from the paper.
    """

    def __init__(self):
        self.num_players = 5
        self.num_assignments = 20

        # Generate all 20 role assignments
        self.assignments = self._generate_assignments()

    def _generate_assignments(self) -> List[Tuple[int, ...]]:
        """Generate all 20 possible role assignments.

        Returns:
            List of tuples (p0_role, p1_role, ..., p4_role)
            Roles: 0=liberal, 1=fascist, 2=hitler
        """
        assignments = []
        from itertools import combinations

        # Choose 3 positions for liberals out of 5
        for lib_positions in combinations(range(5), 3):
            remaining = [i for i in range(5) if i not in lib_positions]
            # Choose which of the 2 remaining is Hitler
            for hitler_pos in range(2):
                assignment = [0] * 5
                for pos in lib_positions:
                    assignment[pos] = 0  # Liberal
                assignment[remaining[hitler_pos]] = 2  # Hitler
                assignment[remaining[1 - hitler_pos]] = 1  # Fascist
                assignments.append(tuple(assignment))

        return assignments

    def update_belief(self, belief: np.ndarray, obs: Dict,
                     strategies: Dict = None) -> np.ndarray:
        """Update belief based on observation.

        According to Algorithm 2 line 19-20:
        - Multiply by reach probabilities (requires strategies)
        - Zero out logically inconsistent assignments

        Args:
            belief: Current belief over 20 assignments
            obs: Observation from environment
            strategies: Player strategies (if available from CFR)

        Returns:
            Updated belief
        """
        new_belief = belief.copy()

        # 1. Apply logical deductions (the part we CAN do)
        new_belief = self._apply_logical_deductions(new_belief, obs)

        # 2. Strategy-based update (requires CFR strategies)
        if strategies is not None:
            # This would implement: b[ρ] * ∏_i π_i(I_i(h, ρ))
            # But we don't have opponent strategies during play
            pass

        # Normalize
        total = np.sum(new_belief)
        if total > 1e-10:
            new_belief /= total
        else:
            # All assignments became impossible (shouldn't happen)
            new_belief = np.ones(20) / 20

        return new_belief

    def _apply_logical_deductions(self, belief: np.ndarray, obs: Dict) -> np.ndarray:
        """Apply logical deductions to zero impossible assignments.

        This implements the (1 - 1{h ⊢ ¬ρ}) term from the paper.
        """
        new_belief = belief.copy()

        # 1. If we know our own role, eliminate inconsistent assignments
        if 'role' in obs and 'player_idx' in obs:
            my_idx = obs['player_idx']
            my_role = obs['role']

            for i, assignment in enumerate(self.assignments):
                if assignment[my_idx] != my_role:
                    new_belief[i] = 0

        # 2. Hitler chancellor deduction (if 3+ fascist policies)
        if obs.get('fasc_policies', 0) >= 3:
            # If someone was chancellor and game didn't end, they're not Hitler
            if 'hist_chancellor' in obs:
                # Check if game ended (all terminations True)
                game_ended = False
                if 'terminations' in obs and obs['terminations']:
                    game_ended = all(obs['terminations'].values())

                if not game_ended:
                    for chanc_idx in obs['hist_chancellor']:
                        if chanc_idx >= 0:
                            for i, assignment in enumerate(self.assignments):
                                if assignment[chanc_idx] == 2:  # Hitler
                                    new_belief[i] = 0

        # 3. Execution deductions
        if 'executed' in obs:
            executed_players = [i for i, x in enumerate(obs['executed']) if x]

            # Check if game ended (Hitler was executed)
            game_ended = all(obs.get('terminations', {}).values()) if 'terminations' in obs else False

            for exec_idx in executed_players:
                if game_ended:
                    # Executed player WAS Hitler
                    for i, assignment in enumerate(self.assignments):
                        if assignment[exec_idx] != 2:
                            new_belief[i] = 0
                else:
                    # Executed player was NOT Hitler
                    for i, assignment in enumerate(self.assignments):
                        if assignment[exec_idx] == 2:
                            new_belief[i] = 0

        # 4. Mission failure deductions
        # If a mission fails, at least one spy was on it
        # If a mission succeeds, either no spies or they chose not to fail
        # This is complex without knowing strategies, so we skip it

        return new_belief


def integrate_with_cfr(belief: np.ndarray, reach_probs: np.ndarray,
                       assignments: List[Tuple]) -> np.ndarray:
    """Integrate belief updates with CFR reach probabilities.

    This would implement the full update from Algorithm 2:
    bterm[ρ] = b[ρ] * ∏_i π_i(I_i(h, ρ))

    Args:
        belief: Current belief
        reach_probs: Reach probabilities from CFR (5, 20)
        assignments: Role assignments

    Returns:
        Updated belief
    """
    new_belief = belief.copy()

    # For each assignment, multiply by reach probabilities
    for i in range(len(assignments)):
        # Product of reach probabilities for all players
        reach_product = 1.0
        for player_idx in range(5):
            reach_product *= reach_probs[player_idx, i]

        new_belief[i] *= reach_product

    # Normalize
    total = np.sum(new_belief)
    if total > 1e-10:
        new_belief /= total

    return new_belief


# The fundamental issue:
#
# DeepRole's belief update REQUIRES the strategies computed during CFR.
# During training/self-play, we have these strategies.
# During actual play against unknown opponents, we DON'T have their strategies.
#
# Options:
# 1. Use only logical deductions (what we implemented above)
# 2. Assume opponents play uniformly random
# 3. Use our own strategy as a proxy (biased)
# 4. Learn opponent models online (complex)
#
# The paper doesn't fully address this gap between training and deployment.