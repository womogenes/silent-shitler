"""Neural network architecture for DeepRole value approximation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, num_players=5, num_assignments=20, hidden_size=80):
        super().__init__()
        self.num_players = num_players
        self.num_assignments = num_assignments

        input_size = num_players + num_assignments

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.win_prob = nn.Linear(hidden_size, num_assignments)

        self.register_buffer('utility_matrix', self._compute_utility_matrix())
        self.register_buffer('infoset_matrix', self._compute_infoset_matrix())

    def _compute_utility_matrix(self):
        from .role_assignments import RoleAssignmentManager
        manager = RoleAssignmentManager()

        util = torch.zeros(self.num_players, 3, self.num_assignments)
        for a_idx, assignment in enumerate(manager.assignments):
            for p_idx in range(self.num_players):
                role = assignment[p_idx]
                util[p_idx, role, a_idx] = 1 if role == 0 else -1
        return util

    def _compute_infoset_matrix(self):
        from .role_assignments import RoleAssignmentManager
        manager = RoleAssignmentManager()

        mat = torch.zeros(self.num_players, 3, self.num_assignments)
        for a_idx, assignment in enumerate(manager.assignments):
            for p_idx in range(self.num_players):
                role = assignment[p_idx]
                mat[p_idx, role, a_idx] = 1
        return mat

    def forward(self, president_idx, belief):
        """
        president_idx: [B]
        belief:        [B, num_assignments]
        Returns:
            values: [B, num_players, num_assignments]
        """
        B = belief.size(0)

        # One-hot presidents: [B, num_players]
        president_oh = F.one_hot(president_idx, self.num_players).float()

        # Input: [B, 25]
        x = torch.cat([president_oh, belief], dim=1)

        # Hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Win probability per assignment: [B, num_assignments]
        win_probs = torch.sigmoid(self.win_prob(x))

        # Liberal value: 2p - 1
        # Fascist/Hitler value: 1 - 2p
        liberal_val  = (2 * win_probs - 1)              # [B, A]
        fascist_val  = (1 - 2 * win_probs)              # [B, A]

        # Construct values using infoset_matrix
        # infoset_matrix: [P, 3, A]
        # roles: 0=lib, 1=fas, 2=hit

        # Expand belief dims for broadcasting
        liberal_val_exp  = liberal_val.unsqueeze(1)     # [B, 1, A]
        fascist_val_exp  = fascist_val.unsqueeze(1)     # [B, 1, A]

        # Mask where each (player, role, assignment) applies
        # masks: [1, P, 3, A]
        masks = self.infoset_matrix.unsqueeze(0)

        # Player values per role:
        # liberal:  role 0
        # fascist: roles 1 and 2 (same payoff)
        values = (
            liberal_val_exp  * masks[:, :, 0, :] +
            fascist_val_exp  * masks[:, :, 1, :] +
            fascist_val_exp  * masks[:, :, 2, :]
        )  # shape [B, P, A]

        return values


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
        import pickle

        # Try torch.load first
        try:
            state = torch.load(path, map_location='cpu')
            for key, state_dict in state.items():
                lib, fasc = map(int, key.split('_'))
                net = ValueNetwork()
                net.load_state_dict(state_dict)
                self.add_network(lib, fasc, net)
            # print(f"Loaded {len(self.networks)} networks from torch file")
            return
        except Exception as e:
            pass  # Try pickle next

        # Try pickle.load as fallback
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)

            # Handle different formats
            if 'networks' in state:
                # New format with networks dict
                for key, net in state['networks'].items():
                    if isinstance(key, tuple) and len(key) == 2:
                        self.add_network(key[0], key[1], net)
                print(f"Loaded {len(self.networks)} networks from pickle file")
            else:
                # Old format with string keys
                for key, state_dict in state.items():
                    lib, fasc = map(int, key.split('_'))
                    net = ValueNetwork()
                    net.load_state_dict(state_dict)
                    self.add_network(lib, fasc, net)
                print(f"Loaded {len(self.networks)} networks from pickle (old format)")
        except Exception as e:
            print(f"Failed to load networks from {path}: {e}")
            self.networks = {}
