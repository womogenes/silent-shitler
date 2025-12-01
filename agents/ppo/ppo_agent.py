"""PPO Agent implementation."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ppo.networks import ActorCritic
from agents.ppo.buffer import RolloutBuffer, EpisodeBuffer
from agents.ppo.observation import ObservationProcessor


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Implements the clipped surrogate objective PPO algorithm.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dims=[128, 128],
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Create actor-critic network
        self.policy = ActorCritic(obs_dim, action_dim, hidden_dims).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Observation processor
        self.obs_processor = ObservationProcessor()

    def get_action(self, obs, action_mask, deterministic=False):
        """
        Get action from policy (for rollout collection).

        Args:
            obs: Observation array
            action_mask: Valid action mask
            deterministic: Whether to take argmax action

        Returns:
            action: Action to take
            log_prob: Log probability of action
            value: Value estimate
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, _ = self.policy.get_action(
                obs_tensor, mask_tensor, deterministic
            )

        return action.item(), log_prob.item(), value.item()

    def collect_rollouts(self, env, n_steps):
        """
        Collect rollouts by playing games.

        Args:
            env: Game environment
            n_steps: Minimum number of steps to collect

        Returns:
            rollout_buffer: Filled RolloutBuffer
        """
        rollout_buffer = RolloutBuffer(
            n_steps, self.obs_dim, self.action_dim, self.gamma, self.gae_lambda
        )

        steps_collected = 0

        while steps_collected < n_steps:
            # Play one full episode
            episode_buffer = self._play_episode(env)

            # Transfer episode to rollout buffer
            episode_buffer.flush_to_rollout_buffer(rollout_buffer)

            steps_collected += len(episode_buffer)

        # Compute advantages using GAE
        rollout_buffer.compute_returns_and_advantages()

        return rollout_buffer

    def _play_episode(self, env):
        """
        Play a single episode with all 5 agents using the current policy.

        Args:
            env: Game environment

        Returns:
            episode_buffer: EpisodeBuffer with episode data
        """
        episode_buffer = EpisodeBuffer(self.obs_dim, self.action_dim)

        env.reset()

        # Track rewards for each agent (sparse, only at end)
        agent_rewards = {agent: 0.0 for agent in env.agents}

        while not all(env.terminations.values()):
            agent = env.agent_selection

            # Get observation and process it
            obs_dict = env.observe(agent)
            obs = self.obs_processor.process(obs_dict)

            # Get valid action mask
            action_mask = self.obs_processor.get_action_mask(env, agent)

            # Get action from policy
            action, log_prob, value = self.get_action(obs, action_mask)

            # Step environment
            env.step(action)

            # Check if episode ended and get reward
            done = env.terminations[agent]
            reward = env.rewards[agent] if done else 0.0

            # Store transition
            episode_buffer.add(obs, action, reward, value, log_prob, done, action_mask)

        return episode_buffer

    def train(self, rollout_buffer, n_epochs=10, batch_size=64):
        """
        Update policy using PPO objective.

        Args:
            rollout_buffer: RolloutBuffer with collected data
            n_epochs: Number of epochs to train on data
            batch_size: Minibatch size

        Returns:
            dict: Training statistics
        """
        # Get all data from buffer
        data = rollout_buffer.get()

        dataset_size = len(data["observations"])

        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get minibatch
                obs_batch = data["observations"][batch_indices].to(self.device)
                actions_batch = data["actions"][batch_indices].to(self.device)
                old_log_probs_batch = data["old_log_probs"][batch_indices].to(self.device)
                advantages_batch = data["advantages"][batch_indices].to(self.device)
                returns_batch = data["returns"][batch_indices].to(self.device)
                masks_batch = data["action_masks"][batch_indices].to(self.device)

                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs_batch, actions_batch, masks_batch
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(values, returns_batch)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                with torch.no_grad():
                    approx_kl = (old_log_probs_batch - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(-entropy_loss.item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(clip_fraction)

        # Average statistics
        return {k: np.mean(v) for k, v in stats.items()}

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
