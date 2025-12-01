"""Neural network architectures for PPO agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared trunk.

    Basic PPO uses shared feature extraction with separate heads for policy and value.
    """

    def __init__(self, obs_dim, action_dim, hidden_dims=[128, 128]):
        """
        Args:
            obs_dim: Dimension of observation vector
            action_dim: Maximum number of actions (we'll mask invalid ones)
            hidden_dims: List of hidden layer sizes
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Policy head (actor)
        self.policy_head = nn.Linear(in_dim, action_dim)

        # Value head (critic)
        self.value_head = nn.Linear(in_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization as commonly used in PPO."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        # Policy head with smaller gain for stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

    def forward(self, obs, action_mask=None):
        """
        Forward pass through both actor and critic.

        Args:
            obs: Observation tensor (batch_size, obs_dim)
            action_mask: Binary mask for valid actions (batch_size, action_dim)

        Returns:
            action_logits: (batch_size, action_dim)
            value: (batch_size, 1)
        """
        features = self.shared_net(obs)

        # Policy logits
        logits = self.policy_head(features)

        # Apply action mask (set invalid actions to very negative logit)
        if action_mask is not None:
            logits = logits + (1 - action_mask) * -1e8

        # Value estimate
        value = self.value_head(features)

        return logits, value

    def get_action(self, obs, action_mask=None, deterministic=False):
        """
        Sample action from policy.

        Args:
            obs: Observation tensor (batch_size, obs_dim)
            action_mask: Binary mask for valid actions (batch_size, action_dim)
            deterministic: If True, take argmax instead of sampling

        Returns:
            action: Sampled actions (batch_size,)
            log_prob: Log probability of actions (batch_size,)
            value: Value estimates (batch_size,)
            entropy: Entropy of action distribution (batch_size,)
        """
        logits, value = self.forward(obs, action_mask)

        # Create categorical distribution
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, value.squeeze(-1), entropy

    def evaluate_actions(self, obs, actions, action_mask=None):
        """
        Evaluate actions taken (used during PPO update).

        Args:
            obs: Observation tensor (batch_size, obs_dim)
            actions: Actions taken (batch_size,)
            action_mask: Binary mask for valid actions (batch_size, action_dim)

        Returns:
            log_prob: Log probability of actions (batch_size,)
            value: Value estimates (batch_size,)
            entropy: Entropy of action distribution (batch_size,)
        """
        logits, value = self.forward(obs, action_mask)

        # Create categorical distribution
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value.squeeze(-1), entropy

    def get_value(self, obs):
        """
        Get value estimate only (faster than full forward pass).

        Args:
            obs: Observation tensor (batch_size, obs_dim)

        Returns:
            value: Value estimates (batch_size,)
        """
        features = self.shared_net(obs)
        value = self.value_head(features)
        return value.squeeze(-1)
