"""Role assignment enumeration for Secret Hitler."""

import numpy as np
from itertools import combinations


class RoleAssignmentManager:
    """Manages the 20 possible role assignments for 5-player Secret Hitler."""

    def __init__(self):
        self.num_players = 5
        self.num_assignments = 20
        self.assignments = self._enumerate_assignments()
        self.assignment_to_idx = {tuple(a): i for i, a in enumerate(self.assignments)}

    def _enumerate_assignments(self):
        """Enumerate all possible role assignments.

        Returns list of 20 assignments, each is array of shape (5,) with values:
        0 = liberal, 1 = fascist, 2 = hitler
        """
        assignments = []
        for lib_indices in combinations(range(5), 3):
            remaining = [i for i in range(5) if i not in lib_indices]
            for fasc_idx in remaining:
                hitler_idx = remaining[1] if remaining[0] == fasc_idx else remaining[0]
                assignment = np.zeros(5, dtype=np.int32)
                assignment[fasc_idx] = 1
                assignment[hitler_idx] = 2
                assignments.append(assignment)
        return np.array(assignments)

    def get_uniform_belief(self):
        """Return uniform belief over all assignments."""
        return np.ones(self.num_assignments) / self.num_assignments

    def get_role_matrix(self, player_idx):
        """Get matrix of which assignments give player_idx each role.

        Returns (3, 20) matrix where row r is 1 if assignment gives player role r.
        """
        matrix = np.zeros((3, self.num_assignments))
        for i, assignment in enumerate(self.assignments):
            role = assignment[player_idx]
            matrix[role, i] = 1
        return matrix

    def get_infoset_indices(self, player_idx, role):
        """Get assignment indices consistent with player having role."""
        return np.where(self.assignments[:, player_idx] == role)[0]