"""Neural network architecture for DeepRole value approximation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNetwork(nn.Module):
    """DeepRole value network with win probability layer.

    Architecture:
    - Input: one-hot president (5) + belief vector (20) = 25 dims
    - Hidden: 2 layers of 80 ReLU units
    - Win probability layer: 20 sigmoid units (one per role assignment)
    - Output: Values for each player's information sets
    """

    def __init__(self, num_players=5, num_assignments=20, hidden_size=80):
        super().__init__()
        self.num_players = num_players
        self.num_assignments = num_assignments

        # Input is one-hot president + belief vector
        input_size = num_players + num_assignments

        # Hidden layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Win probability layer - outputs P(liberals win | assignment)
        self.win_prob = nn.Linear(hidden_size, num_assignments)

        # Precompute utility matrix: u[p,r,a] = utility for player p with role r under assignment a
        # This tells us who wins under each assignment
        self.register_buffer('utility_matrix', self._compute_utility_matrix())

        # Information set mapping matrix: maps (player, assignment) -> information set
        self.register_buffer('infoset_matrix', self._compute_infoset_matrix())

    def _compute_utility_matrix(self):
        """Compute utility matrix for each player under each role and assignment.

        Returns tensor of shape (num_players, 3, num_assignments)
        Entry [p,r,a] = 1 if player p with role r wins under assignment a, else -1
        """
        from .role_assignments import RoleAssignmentManager
        manager = RoleAssignmentManager()

        utility = torch.zeros(self.num_players, 3, self.num_assignments)

        for a_idx, assignment in enumerate(manager.assignments):
            libs_win = True  # We'll determine this based on network output
            for p_idx in range(self.num_players):
                for role in range(3):
                    if assignment[p_idx] == role:
                        # Liberal wins if liberals win, fascist/hitler win if fascists win
                        if role == 0:  # Liberal
                            utility[p_idx, role, a_idx] = 1
                        else:  # Fascist or Hitler
                            utility[p_idx, role, a_idx] = -1

        return utility

    def _compute_infoset_matrix(self):
        """Compute information set mapping.

        Returns tensor of shape (num_players, 3, num_assignments)
        Entry [p,r,a] = 1 if player p has role r in assignment a, else 0
        """
        from .role_assignments import RoleAssignmentManager
        manager = RoleAssignmentManager()

        matrix = torch.zeros(self.num_players, 3, self.num_assignments)

        for a_idx, assignment in enumerate(manager.assignments):
            for p_idx in range(self.num_players):
                role = assignment[p_idx]
                matrix[p_idx, role, a_idx] = 1

        return matrix

    def forward(self, president_idx, belief):
        """Forward pass.

        Args:
            president_idx: Current president (integer 0-4)
            belief: Belief vector over role assignments (20,)

        Returns:
            values: Value matrix of shape (num_players, num_assignments)
                    Entry [p,a] = value for player p under assignment a
        """
        # Prepare input
        president_oh = F.one_hot(president_idx, self.num_players).float()
        x = torch.cat([president_oh, belief])

        # Hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Win probability layer - P(liberals win | assignment)
        win_probs = torch.sigmoid(self.win_prob(x))

        # Convert win probabilities to values
        # For each player and assignment, compute expected value
        values = torch.zeros(self.num_players, self.num_assignments)

        for p_idx in range(self.num_players):
            for a_idx in range(self.num_assignments):
                # Get this player's utility under each role for this assignment
                # We need to know what role player p has in assignment a
                for role in range(3):
                    if self.infoset_matrix[p_idx, role, a_idx] > 0:
                        # Player p has this role in assignment a
                        if role == 0:  # Liberal
                            values[p_idx, a_idx] = 2 * win_probs[a_idx] - 1
                        else:  # Fascist or Hitler
                            values[p_idx, a_idx] = 1 - 2 * win_probs[a_idx]
                        break

        return values

    def compute_infoset_values(self, values, belief):
        """Convert values to information set values.

        Args:
            values: Value matrix (num_players, num_assignments)
            belief: Belief vector (num_assignments,)

        Returns:
            infoset_values: Dict mapping (player, role) -> expected value
        """
        infoset_values = {}

        for p_idx in range(self.num_players):
            for role in range(3):
                # Get assignments where player has this role
                mask = self.infoset_matrix[p_idx, role] > 0
                if mask.any():
                    # Weight by belief over consistent assignments
                    consistent_belief = belief * mask
                    if consistent_belief.sum() > 0:
                        consistent_belief /= consistent_belief.sum()
                        value = (values[p_idx] * consistent_belief).sum()
                        infoset_values[(p_idx, role)] = value.item()

        return infoset_values


class NetworkEnsemble:
    """Manages networks for different game stages."""

    def __init__(self):
        self.networks = {}

    def add_network(self, lib_policies, fasc_policies, network):
        """Add network for game state."""
        key = (lib_policies, fasc_policies)
        self.networks[key] = network

    def get_network(self, lib_policies, fasc_policies):
        """Get network for game state."""
        return self.networks.get((lib_policies, fasc_policies))

    def save(self, path):
        """Save all networks."""
        state = {
            f"{lib}_{fasc}": net.state_dict()
            for (lib, fasc), net in self.networks.items()
        }
        torch.save(state, path)

    def load(self, path):
        """Load networks."""
        state = torch.load(path)
        for key, state_dict in state.items():
            lib, fasc = map(int, key.split('_'))
            net = ValueNetwork()
            net.load_state_dict(state_dict)
            self.add_network(lib, fasc, net)