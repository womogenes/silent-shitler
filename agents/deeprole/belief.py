"""Belief tracking with deductive reasoning for Secret Hitler."""

import numpy as np
from .role_assignments import RoleAssignmentManager


class BeliefTracker:
    """Tracks beliefs over role assignments with deductive reasoning."""

    def __init__(self):
        self.manager = RoleAssignmentManager()

    def update_belief(self, belief, history, observation, strategies):
        """Update belief based on observation and player strategies.

        Args:
            belief: Current belief vector (20,)
            history: Game history dict
            observation: What happened (e.g., vote outcome)
            strategies: Dict mapping player_idx to action probabilities

        Returns:
            Updated belief vector with impossible assignments zeroed
        """
        # Likelihood update from strategies
        new_belief = belief.copy()
        for player_idx, action_probs in strategies.items():
            # Weight each assignment by probability player would take observed action
            for i, assignment in enumerate(self.manager.assignments):
                role = assignment[player_idx]
                # Get the action this player took given observation
                action = self._deduce_action(player_idx, observation, history)
                if action is not None and action in action_probs:
                    new_belief[i] *= action_probs[action]

        # Apply deductive reasoning
        new_belief = self._apply_deductions(new_belief, history, observation)

        # Normalize
        total = np.sum(new_belief)
        if total > 0:
            new_belief /= total
        else:
            # All assignments became impossible - shouldn't happen with correct implementation
            new_belief = self.manager.get_uniform_belief()

        return new_belief

    def _deduce_action(self, player_idx, observation, history):
        """Deduce what action a player took from observation."""
        if 'votes' in observation:
            # Voting phase - observation contains vote outcome
            return observation['votes'].get(player_idx)
        elif 'policy' in observation:
            # Policy played - deduce card discards if this player was involved
            if history['president_idx'] == player_idx:
                return observation.get('president_discard')
            elif history['chancellor_idx'] == player_idx:
                return observation.get('chancellor_discard')
        return None

    def _apply_deductions(self, belief, history, observation):
        """Apply logical deductions to zero impossible assignments."""
        # Hitler chancellor deduction
        if 'chancellor_elected' in observation:
            chanc_idx = observation['chancellor_idx']
            if history['fasc_policies'] >= 3 and not observation.get('game_ended', False):
                # Chancellor can't be Hitler
                for i, assignment in enumerate(self.manager.assignments):
                    if assignment[chanc_idx] == 2:  # Hitler
                        belief[i] = 0

        # Liberal forced play deduction
        if 'policy' in observation and observation['policy'] == 1:  # Fascist played
            if 'president_cards' in observation:
                prez_idx = history['president_idx']
                prez_libs = observation['president_cards']['libs']
                if prez_libs == 3:  # President had all libs, must pass one
                    for i, assignment in enumerate(self.manager.assignments):
                        if assignment[prez_idx] == 0:  # Liberal
                            belief[i] = 0  # Liberal can't discard lib when all 3 are libs

        # Card claim conflicts
        if 'claims' in observation:
            prez_claim = observation['claims'].get('president')
            chanc_claim = observation['claims'].get('chancellor')
            if prez_claim is not None and chanc_claim is not None:
                # Check consistency: chancellor must see subset of president's cards
                min_fascs = max(0, prez_claim['fascs'] - 1)
                max_fascs = min(2, prez_claim['fascs'])
                if chanc_claim['fascs'] < min_fascs or chanc_claim['fascs'] > max_fascs:
                    # Conflict - at least one is lying
                    # This is weaker deduction - we can't eliminate specific assignments
                    # but can track it for suspicion scoring
                    pass

        # Hitler execution
        if 'execution' in observation:
            executed_idx = observation['executed_idx']
            if observation.get('game_ended', False) and observation.get('liberals_win', False):
                # Executed player was Hitler
                for i, assignment in enumerate(self.manager.assignments):
                    if assignment[executed_idx] != 2:
                        belief[i] = 0
            else:
                # Executed player was not Hitler
                for i, assignment in enumerate(self.manager.assignments):
                    if assignment[executed_idx] == 2:
                        belief[i] = 0

        return belief

    def get_infoset_beliefs(self, belief, player_idx, role):
        """Get belief vector conditioned on player having role.

        Returns belief over assignments where player_idx has role.
        """
        indices = self.manager.get_infoset_indices(player_idx, role)
        conditioned = np.zeros_like(belief)
        conditioned[indices] = belief[indices]
        total = np.sum(conditioned)
        if total > 0:
            conditioned /= total
        return conditioned