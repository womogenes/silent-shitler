"""Symmetric MAPPO Agent - both teams train simultaneously."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ppo.mappo_agent import MAPPOAgent
from agents.ppo.mappo_buffer import MAPPORolloutBuffer


class SymmetricMAPPOAgent:
    """
    Symmetric MAPPO training where both teams train simultaneously.

    This is the traditional MAPPO approach:
    - Liberal team has its own MAPPO network
    - Fascist team has its own MAPPO network
    - Both train from the same games (self-play)
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        **mappo_kwargs,
    ):
        """
        Args:
            obs_dim: Local observation dimension (single agent)
            action_dim: Action dimension
            **mappo_kwargs: Arguments passed to both MAPPOAgent instances
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Create separate MAPPO agents for each team
        self.liberal_agent = MAPPOAgent(obs_dim, action_dim, **mappo_kwargs)
        self.fascist_agent = MAPPOAgent(obs_dim, action_dim, **mappo_kwargs)

        print("Symmetric MAPPO: Both teams train simultaneously")

    def collect_rollouts(self, env, n_steps):
        """
        Collect rollouts where both teams use their respective MAPPO policies.

        Args:
            env: Game environment
            n_steps: Minimum number of steps to collect per team

        Returns:
            tuple: (liberal_buffer, fascist_buffer) - One buffer per team
        """
        # Create buffers for both teams
        liberal_buffer = MAPPORolloutBuffer(
            buffer_size=n_steps,
            local_obs_dim=self.obs_dim,
            global_obs_dim=self.obs_dim * 5,
            action_dim=self.action_dim,
            gamma=self.liberal_agent.gamma,
            gae_lambda=self.liberal_agent.gae_lambda,
        )

        fascist_buffer = MAPPORolloutBuffer(
            buffer_size=n_steps,
            local_obs_dim=self.obs_dim,
            global_obs_dim=self.obs_dim * 5,
            action_dim=self.action_dim,
            gamma=self.fascist_agent.gamma,
            gae_lambda=self.fascist_agent.gae_lambda,
        )

        lib_steps = 0
        fasc_steps = 0

        # Collect until both teams have enough steps
        while lib_steps < n_steps or fasc_steps < n_steps:
            lib_added, fasc_added = self._play_episode(
                env, liberal_buffer, fascist_buffer
            )
            lib_steps += lib_added
            fasc_steps += fasc_added

        # Compute advantages for both teams
        liberal_buffer.compute_returns_and_advantages(last_values=0.0)
        fascist_buffer.compute_returns_and_advantages(last_values=0.0)

        return liberal_buffer, fascist_buffer

    def _play_episode(self, env, liberal_buffer, fascist_buffer):
        """
        Play one episode with both teams using their MAPPO policies.

        Args:
            env: Game environment
            liberal_buffer: Buffer for liberal experiences
            fascist_buffer: Buffer for fascist experiences

        Returns:
            tuple: (lib_steps_added, fasc_steps_added)
        """
        env.reset()

        # Track observations for global state construction
        agent_observations = {}

        lib_steps = 0
        fasc_steps = 0

        while not all(env.terminations.values()):
            agent = env.agent_selection

            # Get observation
            obs_dict = env.observe(agent)
            obs_array = self.liberal_agent.obs_processor.process(obs_dict)

            # Store observation for global state
            agent_observations[agent] = obs_array

            # Construct global state
            global_state = self._construct_global_state(agent_observations, env.agents)

            # Determine which team and network to use
            agent_role = env.roles[agent]
            is_liberal = (agent_role == "lib")

            # Select appropriate agent
            mappo_agent = self.liberal_agent if is_liberal else self.fascist_agent
            buffer = liberal_buffer if is_liberal else fascist_buffer

            # Get action mask
            action_mask = mappo_agent.obs_processor.get_action_mask(env, agent)

            # Get action from appropriate policy
            local_obs_t = torch.FloatTensor(obs_array).unsqueeze(0).to(mappo_agent.device)
            global_state_t = torch.FloatTensor(global_state).unsqueeze(0).to(mappo_agent.device)
            action_mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(mappo_agent.device)

            with torch.no_grad():
                action, log_prob, value, _ = mappo_agent.network.get_action(
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

            # Store transition in appropriate buffer
            buffer.add(
                local_obs=obs_array,
                global_state=global_state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=action_mask,
            )

            # Count steps
            if is_liberal:
                lib_steps += 1
            else:
                fasc_steps += 1

        return lib_steps, fasc_steps

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

    def train(self, liberal_buffer, fascist_buffer, n_epochs=10, batch_size=64):
        """
        Train both MAPPO policies.

        Args:
            liberal_buffer: MAPPORolloutBuffer with liberal experiences
            fascist_buffer: MAPPORolloutBuffer with fascist experiences
            n_epochs: Number of epochs to train
            batch_size: Minibatch size

        Returns:
            dict: Training statistics for both teams
        """
        # Train both agents
        lib_stats = self.liberal_agent.train(liberal_buffer, n_epochs, batch_size)
        fasc_stats = self.fascist_agent.train(fascist_buffer, n_epochs, batch_size)

        # Combine statistics
        return {
            "liberal": lib_stats,
            "fascist": fasc_stats,
        }

    def save(self, liberal_path, fascist_path):
        """Save both model checkpoints."""
        self.liberal_agent.save(liberal_path)
        self.fascist_agent.save(fascist_path)

    def load(self, liberal_path, fascist_path):
        """Load both model checkpoints."""
        self.liberal_agent.load(liberal_path)
        self.fascist_agent.load(fascist_path)

    def get_action(self, obs, action_mask, role, deterministic=False):
        """
        Get action from appropriate policy based on role.

        Args:
            obs: Local observation array
            action_mask: Action mask array
            role: Agent role ("lib", "fasc", or "hitty")
            deterministic: If True, take argmax

        Returns:
            tuple: (action, log_prob, value)
        """
        is_liberal = (role == "lib")
        agent = self.liberal_agent if is_liberal else self.fascist_agent
        return agent.get_action(obs, action_mask, deterministic)
