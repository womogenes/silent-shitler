"""Full belief update implementation with CFR strategies.

This implements Algorithm 2 line 19: bterm[ρ] = b[ρ] * ∏_i π_i(I_i(h, ρ))
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import copy


class StrategicBeliefUpdater:
    """Updates beliefs using strategies computed by CFR."""

    def __init__(self):
        self.num_players = 5
        self.num_assignments = 20
        self.assignments = self._generate_assignments()

        # Track game history for computing reach probabilities
        self.action_history = []  # List of (phase, acting_players, actions_taken)

    def _generate_assignments(self) -> List[Tuple[int, ...]]:
        """Generate all 20 possible role assignments."""
        assignments = []
        from itertools import combinations

        for lib_positions in combinations(range(5), 3):
            remaining = [i for i in range(5) if i not in lib_positions]
            for hitler_pos in range(2):
                assignment = [0] * 5
                for pos in lib_positions:
                    assignment[pos] = 0  # Liberal
                assignment[remaining[hitler_pos]] = 2  # Hitler
                assignment[remaining[1 - hitler_pos]] = 1  # Fascist
                assignments.append(tuple(assignment))

        return assignments

    def update_belief_with_strategies(
        self,
        belief: np.ndarray,
        obs: Dict,
        strategies: Dict[str, Dict[int, float]],
        env_state: Any = None
    ) -> np.ndarray:
        """Update belief using CFR-computed strategies.

        This implements: b[ρ] = b[ρ] * ∏_i π_i(I_i(h, ρ))

        Args:
            belief: Current belief over 20 assignments
            obs: Current observation
            strategies: CFR-computed strategies {infoset_key: {action: prob}}
            env_state: Current environment state for context

        Returns:
            Updated belief
        """
        new_belief = belief.copy()

        # For each assignment, compute reach probabilities
        for assign_idx, assignment in enumerate(self.assignments):
            if new_belief[assign_idx] == 0:
                continue  # Skip already impossible assignments

            # Compute product of reach probabilities
            reach_product = 1.0

            # For each player, compute their reach probability
            for player_idx in range(self.num_players):
                role = assignment[player_idx]

                # Get this player's information set given their role
                infoset_key = self._get_infoset_for_player(
                    player_idx, role, obs, env_state, assignment
                )

                # Look up their strategy
                if infoset_key in strategies:
                    strategy = strategies[infoset_key]

                    # Get the action this player took (from history)
                    action_taken = self._get_last_action_for_player(
                        player_idx, obs
                    )

                    if action_taken is not None:
                        # Get probability of this action in their strategy
                        action_prob = strategy.get(action_taken, 0.0)

                        # Multiply reach probability
                        reach_product *= action_prob

                        # If action has 0 probability, this assignment is impossible
                        if action_prob == 0:
                            new_belief[assign_idx] = 0
                            break

            # Update belief by reach probability product
            new_belief[assign_idx] *= reach_product

        # Apply logical consistency checks
        new_belief = self._apply_logical_deductions(new_belief, obs)

        # Normalize
        total = np.sum(new_belief)
        if total > 1e-10:
            new_belief /= total
        else:
            # All became impossible - shouldn't happen
            new_belief = np.ones(20) / 20

        return new_belief

    def _get_infoset_for_player(
        self,
        player_idx: int,
        role: int,
        obs: Dict,
        env_state: Any,
        assignment: Tuple = None
    ) -> str:
        """Construct information set key for a player given their role.

        This matches how vector_cfr.py creates infoset keys.
        """
        # Construct infoset key similar to vector_cfr._get_infoset_key
        parts = []

        # Add role
        role_names = ['lib', 'fasc', 'hitler']
        parts.append(role_names[role])

        # Add game state
        lib_policies = obs.get('lib_policies', 0)
        fasc_policies = obs.get('fasc_policies', 0)
        parts.append(f"{lib_policies}L{fasc_policies}F")

        # Add phase
        phase = obs.get('phase', 'unknown')
        parts.append(phase)

        # Add player index
        parts.append(f"P{player_idx}")

        # For fascists/Hitler, add who they know is on their team
        if role in [1, 2] and assignment:  # Fascist or Hitler
            # Find other fascist team member in the current assignment
            for other_idx, other_role in enumerate(assignment):
                if other_idx != player_idx and other_role in [1, 2]:
                    parts.append(f"knows_P{other_idx}")
                    break

        # Add cards in hand if in card selection phase
        if phase in ['prez_cardsel', 'chanc_cardsel']:
            if 'prez_cards' in obs:
                cards = obs['prez_cards']
                parts.append(f"cards_{cards}")
            elif 'chanc_cards' in obs:
                cards = obs['chanc_cards']
                parts.append(f"cards_{cards}")

        return '_'.join(parts)

    def _get_last_action_for_player(
        self,
        player_idx: int,
        obs: Dict
    ) -> Any:
        """Get the last action taken by a player from observation history."""
        phase = obs.get('phase', '')

        # Check voting history
        if 'hist_votes' in obs and len(obs['hist_votes']) > 0:
            last_votes = obs['hist_votes'][-1]
            if player_idx < len(last_votes):
                return last_votes[player_idx]  # 0 or 1

        # Check nomination history
        if phase == 'nomination' and 'hist_chancellor' in obs:
            if len(obs['hist_chancellor']) > 0:
                return obs['hist_chancellor'][-1]

        # Check policy played
        if 'hist_policy' in obs and len(obs['hist_policy']) > 0:
            # This is trickier - need to know who was in government
            pass

        # Check execution
        if 'executed' in obs:
            for idx, executed in enumerate(obs['executed']):
                if executed:
                    return idx  # The executed player

        return None

    def _apply_logical_deductions(self, belief: np.ndarray, obs: Dict) -> np.ndarray:
        """Apply logical consistency checks (same as before)."""
        new_belief = belief.copy()

        # 1. Own role constraint
        if 'role' in obs and 'player_idx' in obs:
            my_idx = obs['player_idx']
            my_role = obs['role']

            for i, assignment in enumerate(self.assignments):
                if assignment[my_idx] != my_role:
                    new_belief[i] = 0

        # 2. Hitler chancellor deduction
        if obs.get('fasc_policies', 0) >= 3:
            if 'hist_chancellor' in obs:
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

        return new_belief

    def record_action(self, phase: str, player_idx: int, action: Any):
        """Record an action taken for future belief updates."""
        self.action_history.append((phase, player_idx, action))