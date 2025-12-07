"""MAPPO Rollout buffer that stores both local observations and global state."""

import numpy as np
import torch


class MAPPORolloutBuffer:
    """
    Buffer for MAPPO that stores both local observations and global state.

    Key difference from standard buffer:
    - Stores local_obs for actor
    - Stores global_state for critic
    """

    def __init__(self, buffer_size, local_obs_dim, global_obs_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        """
        Args:
            buffer_size: Maximum number of transitions
            local_obs_dim: Local observation dimension
            global_obs_dim: Global state dimension
            action_dim: Action dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage arrays
        self.local_observations = np.zeros((buffer_size, local_obs_dim), dtype=np.float32)
        self.global_states = np.zeros((buffer_size, global_obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, action_dim), dtype=np.float32)

        # Computed during finalize()
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, local_obs, global_state, action, reward, value, log_prob, done, action_mask):
        """
        Add a transition to the buffer.

        Args:
            local_obs: Local observation (local_obs_dim,)
            global_state: Global state (global_obs_dim,)
            action: Action taken (scalar)
            reward: Reward received (scalar)
            value: Value estimate (scalar)
            log_prob: Log probability of action (scalar)
            done: Whether episode terminated (bool)
            action_mask: Valid action mask (action_dim,)
        """
        self.local_observations[self.pos] = local_obs
        self.global_states[self.pos] = global_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)
        self.action_masks[self.pos] = action_mask

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_values=None):
        """
        Compute advantages using GAE.

        Args:
            last_values: Value estimates for states after episode ends
        """
        size = self.buffer_size if self.full else self.pos

        last_gae_lam = 0
        for step in reversed(range(size)):
            if step == size - 1:
                if self.dones[step]:
                    next_value = 0.0
                else:
                    next_value = last_values if last_values is not None else 0.0
            else:
                next_value = self.values[step + 1]

            # TD error
            delta = self.rewards[step] + self.gamma * next_value * (1 - self.dones[step]) - self.values[step]

            # GAE
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * last_gae_lam

            self.advantages[step] = last_gae_lam

        # Returns are advantages + values
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get(self):
        """
        Get all data from buffer.

        Returns:
            Dictionary with all buffer contents as torch tensors
        """
        size = self.buffer_size if self.full else self.pos

        # Normalize advantages
        advantages = self.advantages[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "local_observations": torch.FloatTensor(self.local_observations[:size]),
            "global_states": torch.FloatTensor(self.global_states[:size]),
            "actions": torch.LongTensor(self.actions[:size]),
            "old_log_probs": torch.FloatTensor(self.log_probs[:size]),
            "advantages": torch.FloatTensor(advantages),
            "returns": torch.FloatTensor(self.returns[:size]),
            "action_masks": torch.FloatTensor(self.action_masks[:size]),
        }

    def clear(self):
        """Reset buffer to empty state."""
        self.pos = 0
        self.full = False

    def size(self):
        """Return current number of transitions in buffer."""
        return self.buffer_size if self.full else self.pos
