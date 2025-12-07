"""MAPPO (Multi-Agent PPO) agent implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.ppo.mappo_networks import MAPPOActorCritic
from agents.ppo.mappo_buffer import MAPPORolloutBuffer
from agents.ppo.observation import ObservationProcessor


class MAPPOAgent:
    """
    MAPPO agent with centralized critic.

    Key differences from standard PPO:
    - Critic sees global state (all agents' observations)
    - Actor still sees only local observation
    - This enables better value estimation during training
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dims=[128, 128],
        lr=3e-4,
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
            obs_dim: Local observation dimension (single agent)
            action_dim: Action space size
            hidden_dims: Hidden layer sizes
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            device: Device to use
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

        # Global state dimension (5 agents * local obs dim)
        self.global_obs_dim = obs_dim * 5

        # Network and optimizer
        self.network = MAPPOActorCritic(
            local_obs_dim=obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Observation processor
        self.obs_processor = ObservationProcessor()

    def get_action(self, obs, action_mask, deterministic=False):
        """
        Get action from policy.

        Args:
            obs: Local observation array (obs_dim,)
            action_mask: Action mask array (action_dim,)
            deterministic: If True, take argmax

        Returns:
            action: Selected action (scalar)
            log_prob: Log probability (scalar)
            value: Value estimate (scalar) - NOTE: This is NOT accurate without global state!
        """
        with torch.no_grad():
            local_obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

            # For single agent inference, we don't have global state
            # So we'll just use local obs repeated (not ideal but works for eval)
            dummy_global_state = local_obs.repeat(1, 5)

            action, log_prob, value, _ = self.network.get_action(
                local_obs, dummy_global_state, action_mask_t, deterministic
            )

            return action.item(), log_prob.item(), value.item()

    def collect_rollouts(self, env, num_steps):
        """
        Collect rollouts using current policy.

        Key MAPPO difference: Constructs global state from all agents' observations.

        Args:
            env: Game environment
            num_steps: Number of timesteps to collect

        Returns:
            buffer: Filled MAPPORolloutBuffer
            episode_info: Dict with episode statistics
        """
        buffer = MAPPORolloutBuffer(
            buffer_size=num_steps,
            local_obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        episode_rewards = []
        episode_lengths = []
        current_episode_rewards = []

        # Track observations from all agents to build global state
        # We'll store observations for the current timestep
        agent_observations = {}

        obs, _ = env.reset()

        for step in range(num_steps):
            # Process current agent's local observation
            current_agent = env.agent_selection
            local_obs_array = self.obs_processor.process(obs)
            action_mask = self.obs_processor.get_action_mask(env, current_agent)

            # Store this agent's observation for global state construction
            agent_observations[current_agent] = local_obs_array

            # Construct global state (concatenate all 5 agents' observations in order)
            # NOTE: We need observations from ALL agents to build global state
            # If we don't have all observations yet, use zeros for missing agents
            global_state = self._construct_global_state(agent_observations, env.agents)

            # Get action from policy
            local_obs_t = torch.FloatTensor(local_obs_array).unsqueeze(0).to(self.device)
            global_state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            action_mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value, _ = self.network.get_action(
                    local_obs_t, global_state_t, action_mask_t, deterministic=False
                )

            action = action.item()
            log_prob = log_prob.item()
            value = value.item()

            # Take action
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            # Add to buffer
            buffer.add(
                local_obs=local_obs_array,
                global_state=global_state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=action_mask,
            )

            # Track rewards
            current_episode_rewards.append(reward)

            # Episode end
            if done:
                episode_rewards.append(sum(current_episode_rewards))
                episode_lengths.append(len(current_episode_rewards))
                current_episode_rewards = []
                agent_observations = {}  # Reset for new episode
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Compute returns and advantages
        # For last value, we need to evaluate the critic on the global state
        if not done:
            local_obs_array = self.obs_processor.process(obs)
            agent_observations[env.agent_selection] = local_obs_array
            global_state = self._construct_global_state(agent_observations, env.agents)
            global_state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                last_value = self.network.get_value(global_state_t).item()
        else:
            last_value = 0.0

        buffer.compute_returns_and_advantages(last_value)

        # Episode info
        episode_info = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
            "num_episodes": len(episode_rewards),
        }

        return buffer, episode_info

    def _construct_global_state(self, agent_observations, agent_list):
        """
        Construct global state by concatenating all agents' observations.

        Args:
            agent_observations: Dict mapping agent_id -> local observation array
            agent_list: List of all agent IDs in the environment

        Returns:
            global_state: Concatenated observations (global_obs_dim,)
        """
        global_state = []

        for agent_id in agent_list:
            if agent_id in agent_observations:
                global_state.append(agent_observations[agent_id])
            else:
                # If we don't have this agent's observation yet, use zeros
                global_state.append(np.zeros(self.obs_dim, dtype=np.float32))

        return np.concatenate(global_state)

    def train(self, buffer, num_epochs=4, batch_size=64):
        """
        Update policy using PPO objective.

        Args:
            buffer: MAPPORolloutBuffer with collected data
            num_epochs: Number of update epochs
            batch_size: Minibatch size

        Returns:
            train_info: Dict with training statistics
        """
        # Get data from buffer
        data = buffer.get()

        local_obs = data["local_observations"].to(self.device)
        global_states = data["global_states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["old_log_probs"].to(self.device)
        advantages = data["advantages"].to(self.device)
        returns = data["returns"].to(self.device)
        action_masks = data["action_masks"].to(self.device)

        # Training statistics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        num_updates = 0

        # Multiple epochs over the data
        for epoch in range(num_epochs):
            # Generate random minibatches
            indices = torch.randperm(len(local_obs))

            for start in range(0, len(local_obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Minibatch data
                batch_local_obs = local_obs[batch_indices]
                batch_global_states = global_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch_local_obs, batch_global_states, batch_actions, batch_action_masks
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_updates += 1

        # Training info
        train_info = {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
        }

        return train_info

    def save(self, path):
        """Save model to disk."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
