"""
MAPPO (Multi-Agent PPO) network architectures.

Key difference from standard PPO:
- Centralized Critic: Sees global state (all agents' observations)
- Decentralized Actor: Sees only local observation (same as PPO)

This is the "Centralized Training, Decentralized Execution" (CTDE) paradigm.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MAPPOActorCritic(nn.Module):
    """
    MAPPO Actor-Critic with centralized critic.

    Actor: Local observation → Actions (decentralized execution)
    Critic: Global state → Value estimate (centralized training)
    """

    def __init__(self, local_obs_dim, global_obs_dim, action_dim, hidden_dims=[128, 128]):
        """
        Args:
            local_obs_dim: Dimension of single agent's observation
            global_obs_dim: Dimension of global state (all agents' observations)
            action_dim: Action space size
            hidden_dims: Hidden layer sizes for both networks
        """
        super().__init__()

        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim

        # Actor Network (takes local observation)
        actor_layers = []
        in_dim = local_obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        self.policy_head = nn.Linear(in_dim, action_dim)

        # Critic Network (takes global state - CENTRALIZED!)
        critic_layers = []
        in_dim = global_obs_dim
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.critic_net = nn.Sequential(*critic_layers)
        self.value_head = nn.Linear(in_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization as in PPO."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        # Policy head with smaller gain
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

    def forward_actor(self, local_obs, action_mask=None):
        """
        Actor forward pass (uses local observation only).

        Args:
            local_obs: Local observation tensor (batch_size, local_obs_dim)
            action_mask: Binary mask for valid actions (batch_size, action_dim)

        Returns:
            action_logits: (batch_size, action_dim)
        """
        features = self.actor_net(local_obs)
        logits = self.policy_head(features)

        # Apply action mask
        if action_mask is not None:
            logits = logits + (1 - action_mask) * -1e8

        return logits

    def forward_critic(self, global_state):
        """
        Critic forward pass (uses global state - centralized!).

        Args:
            global_state: Global state tensor (batch_size, global_obs_dim)

        Returns:
            value: Value estimate (batch_size, 1)
        """
        features = self.critic_net(global_state)
        value = self.value_head(features)
        return value

    def get_action(self, local_obs, global_state, action_mask=None, deterministic=False):
        """
        Sample action from policy.

        Args:
            local_obs: Local observation (batch_size, local_obs_dim)
            global_state: Global state (batch_size, global_obs_dim)
            action_mask: Valid action mask (batch_size, action_dim)
            deterministic: If True, take argmax

        Returns:
            action: Sampled actions (batch_size,)
            log_prob: Log probability of actions (batch_size,)
            value: Value estimates (batch_size,)
            entropy: Entropy of action distribution (batch_size,)
        """
        # Actor uses local obs
        logits = self.forward_actor(local_obs, action_mask)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Critic uses global state
        value = self.forward_critic(global_state).squeeze(-1)

        return action, log_prob, value, entropy

    def evaluate_actions(self, local_obs, global_state, actions, action_mask=None):
        """
        Evaluate actions (used during PPO update).

        Args:
            local_obs: Local observations (batch_size, local_obs_dim)
            global_state: Global states (batch_size, global_obs_dim)
            actions: Actions taken (batch_size,)
            action_mask: Valid action masks (batch_size, action_dim)

        Returns:
            log_prob: Log probability of actions (batch_size,)
            value: Value estimates (batch_size,)
            entropy: Entropy of action distribution (batch_size,)
        """
        # Actor evaluation
        logits = self.forward_actor(local_obs, action_mask)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # Critic evaluation
        value = self.forward_critic(global_state).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, global_state):
        """
        Get value estimate only (uses global state).

        Args:
            global_state: Global state tensor (batch_size, global_obs_dim)

        Returns:
            value: Value estimates (batch_size,)
        """
        return self.forward_critic(global_state).squeeze(-1)
