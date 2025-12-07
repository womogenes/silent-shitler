"""Rollout buffer for storing and processing PPO experience."""

import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing rollout data and computing advantages using GAE.
    """

    def __init__(self, buffer_size, obs_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        """
        Args:
            buffer_size: Maximum number of transitions to store
            obs_dim: Observation dimension
            action_dim: Action dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.phase_idxs = np.zeros(buffer_size, dtype=np.int64)  # Phase index for multi-head

        # Computed during finalize()
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done, action_mask, phase_idx=0):
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)
        # Pad mask to action_dim size
        padded_mask = np.zeros(self.action_dim, dtype=np.float32)
        padded_mask[:len(action_mask)] = action_mask
        self.action_masks[self.pos] = padded_mask
        self.phase_idxs[self.pos] = phase_idx

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_values=None):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        GAE formula:
            δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            A_t = δ_t + γλ * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...

        Args:
            last_values: Value estimates for states after episode ends (for bootstrapping)
                        Shape: (n_envs,) or None if all episodes terminated
        """
        # Use actual buffer size
        size = self.buffer_size if self.full else self.pos

        last_gae_lam = 0
        for step in reversed(range(size)):
            if step == size - 1:
                # Last step in buffer
                if self.dones[step]:
                    next_value = 0.0
                else:
                    # Bootstrap from last_values if provided
                    next_value = last_values if last_values is not None else 0.0
            else:
                next_value = self.values[step + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + self.gamma * next_value * (1 - self.dones[step]) - self.values[step]

            # GAE: A_t = δ_t + γλ * A_{t+1} * (1 - done)
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

        # Normalize advantages (common practice in PPO)
        advantages = self.advantages[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "observations": torch.FloatTensor(self.observations[:size]),
            "actions": torch.LongTensor(self.actions[:size]),
            "old_log_probs": torch.FloatTensor(self.log_probs[:size]),
            "advantages": torch.FloatTensor(advantages),
            "returns": torch.FloatTensor(self.returns[:size]),
            "action_masks": torch.FloatTensor(self.action_masks[:size]),
            "phase_idxs": torch.LongTensor(self.phase_idxs[:size]),
        }

    def clear(self):
        """Reset buffer to empty state."""
        self.pos = 0
        self.full = False

    def size(self):
        """Return current number of transitions in buffer."""
        return self.buffer_size if self.full else self.pos


class EpisodeBuffer:
    """
    Temporary buffer for collecting a single episode's data.

    In multi-agent turn-based games, we collect full episodes then add to RolloutBuffer.
    """

    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Lists to store episode data (variable length)
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
        self.phase_idxs = []

    def add(self, obs, action, reward, value, log_prob, done, action_mask, phase_idx=0):
        """Add transition to episode buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.phase_idxs.append(phase_idx)

    def flush_to_rollout_buffer(self, rollout_buffer):
        """
        Transfer all episode data to rollout buffer.

        Args:
            rollout_buffer: RolloutBuffer instance
        """
        for i in range(len(self.observations)):
            rollout_buffer.add(
                self.observations[i],
                self.actions[i],
                self.rewards[i],
                self.values[i],
                self.log_probs[i],
                self.dones[i],
                self.action_masks[i],
                self.phase_idxs[i],
            )

    def clear(self):
        """Clear episode buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
        self.phase_idxs = []

    def __len__(self):
        return len(self.observations)
