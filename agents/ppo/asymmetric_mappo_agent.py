"""Asymmetric MAPPO Agent - trains one team while other team uses random policy."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ppo.mappo_agent import MAPPOAgent
from agents.ppo.mappo_buffer import MAPPORolloutBuffer
from agents.random_agent import RandomAgent


class AsymmetricMAPPOAgent:
    """
    Asymmetric MAPPO training where one team trains and the other uses random policy.

    Key MAPPO feature: Uses centralized critic (sees global state) during training,
    but decentralized actor (sees local obs) during execution.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        train_team="liberal",  # "liberal" or "fascist"
        **mappo_kwargs,
    ):
        """
        Args:
            obs_dim: Local observation dimension (single agent)
            action_dim: Action dimension
            train_team: Which team to train ("liberal" or "fascist")
            **mappo_kwargs: Arguments passed to MAPPOAgent
        """
        self.train_team = train_team
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Create MAPPO agent for the team we're training
        self.mappo_agent = MAPPOAgent(obs_dim, action_dim, **mappo_kwargs)

        # Random agent for the other team
        self.random_agent = RandomAgent()

        print(f"Asymmetric MAPPO training: {train_team} team trains, other team random")

    def collect_rollouts(self, env, n_steps):
        """
        Collect rollouts where trained team uses MAPPO, other team uses random.

        Key MAPPO difference: Constructs global state from ALL agents' observations.

        Args:
            env: Game environment
            n_steps: Minimum number of steps to collect

        Returns:
            rollout_buffer: Filled MAPPORolloutBuffer (only with trained team's experiences)
        """
        rollout_buffer = MAPPORolloutBuffer(
            buffer_size=n_steps,
            local_obs_dim=self.obs_dim,
            global_obs_dim=self.obs_dim * 5,  # 5 agents
            action_dim=self.action_dim,
            gamma=self.mappo_agent.gamma,
            gae_lambda=self.mappo_agent.gae_lambda,
        )

        steps_collected = 0

        while steps_collected < n_steps:
            # Play one full episode
            episode_steps = self._play_episode(env, rollout_buffer)
            steps_collected += episode_steps

        # Compute advantages using GAE
        # For last value, we'd need the global state, but we'll use 0 for simplicity
        # (episodes are complete so this is fine)
        rollout_buffer.compute_returns_and_advantages(last_values=0.0)

        return rollout_buffer

    def _play_episode(self, env, rollout_buffer):
        """
        Play episode with asymmetric policies, storing trained team's experiences.

        Key MAPPO feature: Tracks ALL agents' observations to build global state.

        Args:
            env: Game environment
            rollout_buffer: Buffer to add experiences to

        Returns:
            int: Number of steps added to buffer (trained team only)
        """
        env.reset()

        # Track observations for all agents to build global state
        # Map: agent_id -> observation array
        agent_observations = {}

        steps_added = 0

        while not all(env.terminations.values()):
            agent = env.agent_selection

            # Get observation
            obs_dict = env.observe(agent)
            obs_array = self.mappo_agent.obs_processor.process(obs_dict)

            # Store this agent's observation for global state
            agent_observations[agent] = obs_array

            # Construct global state from all agents' observations
            global_state = self._construct_global_state(agent_observations, env.agents)

            # Determine which team this agent is on
            agent_role = env.roles[agent]
            is_trained_team = self._is_trained_team(agent_role)

            if is_trained_team:
                # Use MAPPO policy for trained team
                action_mask = self.mappo_agent.obs_processor.get_action_mask(env, agent)

                # Get action from policy
                local_obs_t = torch.FloatTensor(obs_array).unsqueeze(0).to(self.mappo_agent.device)
                global_state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.mappo_agent.device)
                action_mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.mappo_agent.device)

                with torch.no_grad():
                    action, log_prob, value, _ = self.mappo_agent.network.get_action(
                        local_obs_t, global_state_t, action_mask_t, deterministic=False
                    )

                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

                # Step environment
                env.step(action)

                # Get reward (only at end)
                done = env.terminations[agent]
                reward = env.rewards[agent] if done else 0.0

                # Store transition (ONLY for trained team)
                rollout_buffer.add(
                    local_obs=obs_array,
                    global_state=global_state,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    action_mask=action_mask,
                )

                steps_added += 1

            else:
                # Use random policy for other team
                action_space = env.action_space(agent)
                action = self.random_agent.get_action(obs_dict, action_space)

                # Step environment (no storage for random team)
                env.step(action)

        return steps_added

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

    def _is_trained_team(self, role):
        """Check if this role is on the trained team."""
        if self.train_team == "liberal":
            return role == "lib"
        else:  # fascist
            return role in ["fasc", "hitty"]

    def train(self, rollout_buffer, n_epochs=10, batch_size=64):
        """
        Train the MAPPO policy.

        Args:
            rollout_buffer: MAPPORolloutBuffer with collected data
            n_epochs: Number of epochs to train
            batch_size: Minibatch size

        Returns:
            dict: Training statistics
        """
        return self.mappo_agent.train(rollout_buffer, n_epochs, batch_size)

    def save(self, path):
        """Save model checkpoint."""
        self.mappo_agent.save(path)

    def load(self, path):
        """Load model checkpoint."""
        self.mappo_agent.load(path)

    def get_action(self, obs, action_mask, deterministic=False):
        """Get action from trained policy (for evaluation)."""
        return self.mappo_agent.get_action(obs, action_mask, deterministic)
