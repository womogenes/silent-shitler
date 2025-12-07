"""Observation processing for PPO agent."""

import numpy as np
import torch


class ObservationProcessor:
    """Converts game observations to flat feature vectors."""

    def __init__(self, max_history_length=10):
        """
        Args:
            max_history_length: Maximum number of governments to include in history
        """
        self.max_history_length = max_history_length
        self.obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self):
        """Calculate the total observation dimension."""
        # Scalar features
        scalar_dim = 0
        scalar_dim += 3  # role (one-hot: lib, fasc, hitler)
        scalar_dim += 1  # lib_policies (0-5)
        scalar_dim += 1  # fasc_policies (0-6)
        scalar_dim += 1  # election_tracker (0-3)
        scalar_dim += 5  # president_idx (one-hot)
        scalar_dim += 6  # chancellor_nominee (one-hot with -1 as first position)
        scalar_dim += 5  # executed mask
        scalar_dim += 5  # all_roles (raw values: -1, 0, 1, 2)

        # History features (per government)
        # Each government: president(5) + chancellor(5) + votes(5) + succeeded(1) +
        #                  policy(1) + prez_claim(1) + chanc_claim(1) + execution(6)
        history_per_gov = 5 + 5 + 5 + 1 + 1 + 1 + 1 + 6
        history_dim = history_per_gov * self.max_history_length

        return scalar_dim + history_dim

    def process(self, obs):
        """
        Convert observation dict to flat numpy array.

        Args:
            obs: Observation dict from environment

        Returns:
            numpy array of shape (obs_dim,)
        """
        features = []

        # Role (one-hot)
        role_onehot = np.zeros(3)
        role_onehot[obs["role"]] = 1
        features.append(role_onehot)

        # Scalar features
        features.append([obs["lib_policies"] / 5.0])  # Normalize
        features.append([obs["fasc_policies"] / 6.0])
        features.append([obs["election_tracker"] / 3.0])

        # President idx (one-hot)
        prez_onehot = np.zeros(5)
        prez_onehot[obs["president_idx"]] = 1
        features.append(prez_onehot)

        # Chancellor nominee (one-hot with -1 mapped to position 0)
        chanc_onehot = np.zeros(6)
        chanc_onehot[obs["chancellor_nominee"] + 1] = 1
        features.append(chanc_onehot)

        # Executed mask
        features.append(obs["executed"])

        # All roles (normalize to [-1, 1])
        all_roles = np.array(obs["all_roles"]) / 2.0
        features.append(all_roles)

        # Process history (last N governments)
        history = self._process_history(obs)
        features.append(history)

        return np.concatenate(features, dtype=np.float32)

    def _process_history(self, obs):
        """Process game history into fixed-size vector."""
        n_govs = len(obs["hist_president"])

        # Take last max_history_length governments
        start_idx = max(0, n_govs - self.max_history_length)

        history_features = []

        for i in range(self.max_history_length):
            gov_idx = start_idx + i

            if gov_idx < n_govs:
                # This government exists
                # President (one-hot)
                prez_oh = np.zeros(5)
                prez_oh[obs["hist_president"][gov_idx]] = 1

                # Chancellor (one-hot)
                chanc_oh = np.zeros(5)
                chanc_oh[obs["hist_chancellor"][gov_idx]] = 1

                # Votes (5 values: -1, 0, or 1)
                votes = np.array(obs["hist_votes"][gov_idx], dtype=np.float32)

                # Succeeded (0 or 1)
                succeeded = [float(obs["hist_succeeded"][gov_idx])]

                # Policy played (-1, 0, or 1)
                policy = [float(obs["hist_policy"][gov_idx])]

                # President claim (0-3, normalize)
                prez_claim = [float(obs["hist_prez_claim"][gov_idx]) / 3.0]

                # Chancellor claim (0-2, normalize)
                chanc_claim = [float(obs["hist_chanc_claim"][gov_idx]) / 2.0]

                # Execution (one-hot with -1 mapped to position 0)
                exec_oh = np.zeros(6)
                exec_oh[obs["hist_execution"][gov_idx] + 1] = 1

                gov_features = np.concatenate([
                    prez_oh, chanc_oh, votes, succeeded,
                    policy, prez_claim, chanc_claim, exec_oh
                ])
            else:
                # Padding for missing history
                gov_features = np.zeros(5 + 5 + 5 + 1 + 1 + 1 + 1 + 6)

            history_features.append(gov_features)

        return np.concatenate(history_features)

    def get_action_mask(self, env, agent):
        """
        Get valid action mask for current agent.

        Args:
            env: Game environment
            agent: Current agent

        Returns:
            numpy array of shape (max_actions,) with 1 for valid actions, 0 for invalid
        """
        action_space = env.action_space(agent)
        n_valid = action_space.n

        # Create mask (we'll use max_actions=6 to cover all cases)
        max_actions = 6
        mask = np.zeros(max_actions, dtype=np.float32)
        mask[:n_valid] = 1.0

        return mask


def batch_observations(obs_list):
    """
    Convert list of observation arrays to batched tensor.

    Args:
        obs_list: List of numpy arrays

    Returns:
        torch.Tensor of shape (batch_size, obs_dim)
    """
    return torch.FloatTensor(np.array(obs_list))


def batch_masks(mask_list):
    """
    Convert list of action masks to batched tensor.

    Args:
        mask_list: List of numpy arrays

    Returns:
        torch.Tensor of shape (batch_size, max_actions)
    """
    return torch.FloatTensor(np.array(mask_list))
