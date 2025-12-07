"""Asymmetric PPO Agent - trains one team while other team uses random policy."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.ppo.ppo_agent import PPOAgent
from agents.ppo.networks import PHASE_TO_IDX
from agents.ppo.buffer import RolloutBuffer, EpisodeBuffer
from agents.random_agent import RandomAgent


class AsymmetricPPOAgent:
    """
    Asymmetric PPO training where one team trains and the other uses random policy.

    This removes the self-play equilibrium problem by fixing one team's strategy.
    """

    def __init__(self, obs_dim, train_team="liberal", **ppo_kwargs):
        self.train_team = train_team
        self.obs_dim = obs_dim
        self.ppo_agent = PPOAgent(obs_dim, **ppo_kwargs)
        self.random_agent = RandomAgent()
        print(f"Asymmetric training: {train_team} team trains, other team random")

    def collect_rollouts(self, env, n_steps):
        """Collect rollouts where trained team uses PPO, other team uses random."""
        rollout_buffer = RolloutBuffer(
            n_steps, self.obs_dim, 5, self.ppo_agent.gamma, self.ppo_agent.gae_lambda
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
        """Play episode with asymmetric policies."""
        episode_buffer = EpisodeBuffer(self.obs_dim, 5)
        env.reset()

        while not all(env.terminations.values()):
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            agent_role = env.roles[agent]

            if self._is_trained_team(agent_role):
                obs = self.ppo_agent.obs_processor.process(obs_dict)
                phase_idx = PHASE_TO_IDX.get(env.phase, 0)
                action_mask = self.ppo_agent.obs_processor.get_action_mask(env, agent)
                action, log_prob, value = self.ppo_agent.get_action(obs, phase_idx, action_mask)
                env.step(action)

                done = env.terminations[agent]
                reward = env.rewards[agent] if done else 0.0
                episode_buffer.add(obs, action, reward, value, log_prob, done, action_mask, phase_idx)
            else:
                action_space = env.action_space(agent)
                action = self.random_agent.get_action(obs_dict, action_space)
                env.step(action)

        return episode_buffer

    def _is_trained_team(self, role):
        return role == "lib" if self.train_team == "liberal" else role in ["fasc", "hitty"]

    def train(self, rollout_buffer, n_epochs=10, batch_size=64):
        return self.ppo_agent.train(rollout_buffer, n_epochs, batch_size)

    def save(self, path):
        self.ppo_agent.save(path)

    def load(self, path):
        self.ppo_agent.load(path)

    def get_action(self, obs, phase_idx, action_mask, deterministic=False):
        return self.ppo_agent.get_action(obs, phase_idx, action_mask, deterministic)
