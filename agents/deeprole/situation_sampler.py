"""Game situation sampler for DeepRole (Algorithm 4 adapted for Secret Hitler)."""

import numpy as np
from itertools import combinations
from .role_assignments import RoleAssignmentManager


class SituationSampler:
    """Samples diverse game situations for training (Algorithm 4)."""

    def __init__(self):
        self.manager = RoleAssignmentManager()

    def sample_situation(self, lib_policies, fasc_policies):
        """Sample a random game situation from this game part.

        Algorithm 4 from the paper, adapted for Secret Hitler.
        Instead of sampling failed missions, we sample which governments
        enacted which policies.

        Returns:
            president_idx: Current president (0-4)
            belief: Belief distribution over role assignments (20,)
        """
        # Sample which governments passed which policies
        gov_outcomes = self._sample_government_outcomes(lib_policies, fasc_policies)

        # Calculate which fascist teams are consistent with outcomes
        fascist_teams = self._get_consistent_fascist_teams(gov_outcomes)

        # Sample probability distribution over fascist teams using Dirichlet
        # Alpha=1 gives uniform prior, leading to diverse distributions
        if fascist_teams:
            alpha = np.ones(len(fascist_teams))
            team_probs = np.random.dirichlet(alpha)
        else:
            # All teams possible
            fascist_teams = self._get_all_fascist_teams()
            team_probs = np.random.dirichlet(np.ones(len(fascist_teams)))

        # Create belief distribution over all 20 role assignments
        belief = np.zeros(self.manager.num_assignments)
        for team_idx, team in enumerate(fascist_teams):
            team_prob = team_probs[team_idx]
            # Find which assignments correspond to this fascist team
            for i, assignment in enumerate(self.manager.assignments):
                fascist_players = set(np.where(assignment > 0)[0])
                if fascist_players == set(team):
                    # Within a team, Hitler could be either fascist
                    belief[i] = team_prob / 2  # Equal probability for Hitler position

        # Normalize belief
        if belief.sum() > 0:
            belief /= belief.sum()
        else:
            belief = self.manager.get_uniform_belief()

        # Sample president uniformly
        president_idx = np.random.randint(5)

        return president_idx, belief

    def _sample_government_outcomes(self, lib_policies, fasc_policies):
        """Sample which governments enacted which policies.

        Returns list of (president, chancellor, policy) tuples.
        """
        outcomes = []
        total_govs = lib_policies + fasc_policies

        # Randomly assign which governments passed liberal vs fascist
        policies = [0] * lib_policies + [1] * fasc_policies
        np.random.shuffle(policies)

        for i, policy in enumerate(policies):
            # Sample random president and chancellor
            president = np.random.randint(5)
            chancellor = np.random.choice([j for j in range(5) if j != president])
            outcomes.append((president, chancellor, policy))

        return outcomes

    def _get_consistent_fascist_teams(self, gov_outcomes):
        """Get fascist teams consistent with government outcomes.

        For now, we return all possible teams since Secret Hitler doesn't
        have as clear constraints as Avalon's mission failures.
        We could add constraints based on:
        - Liberals wouldn't consistently pass fascist policies
        - Fascists might coordinate to pass policies

        Returns list of fascist teams (each team is tuple of 2 player indices).
        """
        # Count how many fascist policies each player was involved in
        fasc_involvement = [0] * 5
        lib_involvement = [0] * 5

        for prez, chanc, policy in gov_outcomes:
            if policy == 1:  # Fascist
                fasc_involvement[prez] += 1
                fasc_involvement[chanc] += 1
            else:  # Liberal
                lib_involvement[prez] += 1
                lib_involvement[chanc] += 1

        # Heuristic: players with high fascist involvement more likely fascist
        # But don't eliminate any teams entirely (soft constraint)
        all_teams = self._get_all_fascist_teams()

        # Could filter teams here based on involvement patterns
        # For now, return all teams to maintain diversity
        return all_teams

    def _get_all_fascist_teams(self):
        """Get all possible fascist teams (2 players out of 5)."""
        return list(combinations(range(5), 2))


class AdvancedSituationSampler(SituationSampler):
    """Enhanced sampler with more sophisticated belief generation."""

    def sample_situation_with_constraints(self, lib_policies, fasc_policies):
        """Sample situation with deductive constraints applied.

        This version applies logical deductions to make beliefs more realistic:
        - If someone was chancellor after 3 fasc and game didn't end, not Hitler
        - Track deck composition constraints
        """
        president_idx, belief = self.sample_situation(lib_policies, fasc_policies)

        # Apply deductive constraints
        if fasc_policies >= 3:
            # Randomly decide if someone was tested as chancellor
            if np.random.random() < 0.3:
                tested_player = np.random.randint(5)
                # Zero assignments where this player is Hitler
                for i, assignment in enumerate(self.manager.assignments):
                    if assignment[tested_player] == 2:  # Hitler
                        belief[i] = 0

                # Renormalize
                if belief.sum() > 0:
                    belief /= belief.sum()
                else:
                    belief = self.manager.get_uniform_belief()

        return president_idx, belief

    def sample_diverse_situations(self, lib_policies, fasc_policies, n_samples=10):
        """Generate multiple diverse situations for a game state.

        Uses different Dirichlet concentration parameters to ensure diversity:
        - Low concentration (alpha < 1): Sparse beliefs (high confidence)
        - High concentration (alpha > 1): Uniform beliefs (low confidence)
        """
        situations = []

        for i in range(n_samples):
            # Vary concentration parameter for diversity
            if i < n_samples // 3:
                # Sparse beliefs (high confidence)
                concentration = 0.1
            elif i < 2 * n_samples // 3:
                # Moderate beliefs
                concentration = 1.0
            else:
                # Uniform beliefs (high uncertainty)
                concentration = 10.0

            president_idx, belief = self._sample_with_concentration(
                lib_policies, fasc_policies, concentration
            )
            situations.append((president_idx, belief))

        return situations

    def _sample_with_concentration(self, lib_policies, fasc_policies, concentration):
        """Sample situation with specific Dirichlet concentration."""
        gov_outcomes = self._sample_government_outcomes(lib_policies, fasc_policies)
        fascist_teams = self._get_all_fascist_teams()

        # Use concentration parameter for Dirichlet
        alpha = np.ones(len(fascist_teams)) * concentration
        team_probs = np.random.dirichlet(alpha)

        # Create belief distribution
        belief = np.zeros(self.manager.num_assignments)
        for team_idx, team in enumerate(fascist_teams):
            team_prob = team_probs[team_idx]
            for i, assignment in enumerate(self.manager.assignments):
                fascist_players = set(np.where(assignment > 0)[0])
                if fascist_players == set(team):
                    belief[i] = team_prob / 2

        if belief.sum() > 0:
            belief /= belief.sum()
        else:
            belief = self.manager.get_uniform_belief()

        president_idx = np.random.randint(5)
        return president_idx, belief