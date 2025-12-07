"""Asymmetric PPO Agent - trains one team while other team uses random policy."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ppo.ppo_agent import PPOAgent
from agents.ppo.buffer import RolloutBuffer, EpisodeBuffer
from agents.random_agent import RandomAgent


class AsymmetricPPOAgent:
    """
    Asymmetric PPO training where one team trains and the other uses random policy.

    This removes the self-play equilibrium problem by fixing one team's strategy.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        train_team="liberal",  # "liberal" or "fascist"
        **ppo_kwargs,
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            train_team: Which team to train ("liberal" or "fascist")
            **ppo_kwargs: Arguments passed to PPOAgent
        """
        self.train_team = train_team
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Create PPO agent for the team we're training
        self.ppo_agent = PPOAgent(obs_dim, action_dim, **ppo_kwargs)

        # Random agent for the other team
        self.random_agent = RandomAgent()

        print(f"Asymmetric training: {train_team} team trains, other team random")

    def collect_rollouts(self, env, n_steps):
        """
        Collect rollouts where trained team uses PPO, other team uses random.

        Args:
            env: Game environment
            n_steps: Minimum number of steps to collect

        Returns:
            rollout_buffer: Filled RolloutBuffer (only with trained team's experiences)
        """
        rollout_buffer = RolloutBuffer(
            n_steps,
            self.obs_dim,
            self.action_dim,
            self.ppo_agent.gamma,
            self.ppo_agent.gae_lambda,
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
        Play episode with asymmetric policies.

        Args:
            env: Game environment

        Returns:
            episode_buffer: EpisodeBuffer with ONLY trained team's experiences
        """
        episode_buffer = EpisodeBuffer(self.obs_dim, self.action_dim)

        env.reset()

        while not all(env.terminations.values()):
            agent = env.agent_selection

            # Get observation
            obs_dict = env.observe(agent)

            # Determine which team this agent is on
            agent_role = env.roles[agent]
            is_trained_team = self._is_trained_team(agent_role)

            if is_trained_team:
                # Use PPO policy for trained team
                obs = self.ppo_agent.obs_processor.process(obs_dict)
                action_mask = self.ppo_agent.obs_processor.get_action_mask(env, agent)
                action, log_prob, value = self.ppo_agent.get_action(obs, action_mask)

                # Step environment
                env.step(action)

                # Get reward (only at end)
                done = env.terminations[agent]
                reward = env.rewards[agent] if done else 0.0

                # Store transition (ONLY for trained team)
                episode_buffer.add(obs, action, reward, value, log_prob, done, action_mask)

            else:
                # Use random policy for other team
                action_space = env.action_space(agent)
                action = self.random_agent.get_action(obs_dict, action_space)

                # Step environment (no storage for random team)
                env.step(action)

        return episode_buffer

    def _is_trained_team(self, role):
        """Check if this role is on the trained team."""
        if self.train_team == "liberal":
            return role == "lib"
        else:  # fascist
            return role in ["fasc", "hitty"]

    def train(self, rollout_buffer, n_epochs=10, batch_size=64):
        """
        Train the PPO policy.

        Args:
            rollout_buffer: RolloutBuffer with collected data
            n_epochs: Number of epochs to train
            batch_size: Minibatch size

        Returns:
            dict: Training statistics
        """
        return self.ppo_agent.train(rollout_buffer, n_epochs, batch_size)

    def save(self, path):
        """Save model checkpoint."""
        self.ppo_agent.save(path)

    def load(self, path):
        """Load model checkpoint."""
        self.ppo_agent.load(path)

    def get_action(self, obs, action_mask, deterministic=False):
        """Get action from trained policy (for evaluation)."""
        return self.ppo_agent.get_action(obs, action_mask, deterministic)
