"""DeepRole agent v2 - Clean implementation using state serialization."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import pickle

from shitler_env.agent import BaseAgent
from shitler_env.game import ShitlerEnv
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.belief import BeliefTracker
from agents.deeprole.belief_with_strategies import StrategicBeliefUpdater
from agents.deeprole.networks import NetworkEnsemble


class DeepRoleAgentV2(BaseAgent):
    """Clean DeepRole agent using proper state serialization."""

    def __init__(self, networks_path="trained_networks.pkl", cfr_iterations=50, max_depth=3):
        """Initialize DeepRole agent.

        Args:
            networks_path: Path to trained neural networks
            cfr_iterations: Number of CFR iterations for real-time planning
            max_depth: Maximum search depth for CFR
        """
        super().__init__()

        # Load trained networks
        self.networks = NetworkEnsemble()
        if Path(networks_path).exists():
            self.networks.load(networks_path)
            print(f"Loaded networks from {networks_path}")
        else:
            print(f"Warning: No networks found at {networks_path}, using random play")

        # CFR solver
        self.cfr = VectorCFR()
        self.cfr_iterations = cfr_iterations
        self.max_depth = max_depth

        # Belief tracking
        self.belief_tracker = BeliefTracker()
        self.strategic_belief_updater = StrategicBeliefUpdater()

        # Game state
        self.player_idx = None
        self.current_belief = None
        self.stored_strategies = None
        self.last_env_state = None

        # Statistics
        self.action_stats = {
            'cfr_strategy': 0,
            'cfr_fallback': 0,
            'cfr_error': 0,
            'total': 0
        }

    def reset(self, player_idx=None):
        """Reset agent for new game."""
        self.player_idx = player_idx
        self.current_belief = np.ones(20) / 20
        self.stored_strategies = None
        self.last_env_state = None
        self.cfr = VectorCFR()

    def get_action(self, obs, action_space=None, game_state=None, **kwargs):
        """Get action using DeepRole algorithm.

        Args:
            obs: Observation dictionary from environment
            action_space: Action space (optional)
            game_state: Full game state dict if available (much cleaner!)
            **kwargs: Additional arguments like agent_name

        Returns:
            Selected action
        """
        # Initialize if needed
        if self.current_belief is None:
            self.current_belief = np.ones(20) / 20

        # Extract player index
        if self.player_idx is None:
            if 'agent_name' in kwargs and kwargs['agent_name'].startswith('P'):
                self.player_idx = int(kwargs['agent_name'][1])
            else:
                self.player_idx = 0

        # Update belief based on previous strategies if available
        if self.stored_strategies is not None:
            self._update_belief(obs)

        # Get valid actions
        valid_actions = self.get_valid_actions(obs)
        if not valid_actions:
            return 0

        # Get phase
        phase = obs.get("phase", "")

        # Use CFR for all decision phases
        if phase in ["nomination", "execution", "prez_cardsel", "chanc_cardsel",
                     "voting", "prez_claim", "chanc_claim"]:
            return self._get_cfr_action(obs, valid_actions, game_state)

        # Default fallback
        return valid_actions[0] if valid_actions else 0

    def _get_cfr_action(self, obs, valid_actions, game_state=None):
        """Get action using CFR planning.

        This is the CLEAN version that uses state serialization when available.
        """
        # Create or restore environment
        if game_state is not None:
            # CLEAN PATH: Use full state directly
            env = ShitlerEnv.from_state_dict(game_state)
        else:
            # FALLBACK: Try to reconstruct from observation
            # This shouldn't be needed if game_state is provided
            print("Warning: No game_state provided, falling back to reconstruction")
            return np.random.choice(valid_actions) if valid_actions else 0

        # Get neural networks for current state
        network_key = (env.lib_policies, env.fasc_policies)
        neural_nets = None
        if hasattr(self.networks, 'networks') and network_key in self.networks.networks:
            neural_nets = {network_key: self.networks.networks[network_key]}

        # Run CFR
        try:
            # Ensure belief is initialized
            if self.current_belief is None:
                self.current_belief = np.ones(20) / 20

            # Run CFR simulation
            values = self.cfr.solve_situation(
                env,
                self.current_belief,
                num_iterations=self.cfr_iterations,
                averaging_delay=self.cfr_iterations // 3,
                neural_nets=neural_nets,
                max_depth=self.max_depth
            )

            # Get strategies from CFR
            avg_strategies = self.cfr.get_average_strategies()
            self.stored_strategies = avg_strategies

            # Get strategy for current player's information set
            infoset_key = self.cfr._get_infoset_key(env, self.player_idx)

            if infoset_key in avg_strategies:
                strategy = avg_strategies[infoset_key]

                # Filter to valid actions
                valid_strategy = {
                    a: p for a, p in strategy.items()
                    if a in valid_actions
                }

                if valid_strategy:
                    # Normalize probabilities
                    total = sum(valid_strategy.values())
                    if total > 0:
                        for a in valid_strategy:
                            valid_strategy[a] /= total

                        # Select action (90% greedy, 10% stochastic)
                        actions = list(valid_strategy.keys())
                        probs = list(valid_strategy.values())

                        if np.random.random() < 0.9:
                            action = actions[np.argmax(probs)]
                        else:
                            action = np.random.choice(actions, p=probs)

                        self.action_stats['cfr_strategy'] += 1
                        self.action_stats['total'] += 1
                        return action

            # No strategy found
            self.action_stats['cfr_fallback'] += 1
            self.action_stats['total'] += 1
            return np.random.choice(valid_actions) if valid_actions else 0

        except Exception as e:
            print(f"CFR error: {e}")
            self.action_stats['cfr_error'] += 1
            self.action_stats['total'] += 1
            return np.random.choice(valid_actions) if valid_actions else 0

    def _update_belief(self, obs):
        """Update belief state based on observation."""
        if self.current_belief is None or self.stored_strategies is None:
            return

        obs_with_idx = obs.copy()
        obs_with_idx['player_idx'] = self.player_idx

        self.current_belief = self.strategic_belief_updater.update_belief_with_strategies(
            self.current_belief,
            obs_with_idx,
            self.stored_strategies,
            self.last_env_state
        )

    def get_valid_actions(self, obs):
        """Extract valid actions from observation."""
        # Check masks in observation
        for mask_key in ['nomination_mask', 'execution_mask', 'card_action_mask']:
            if mask_key in obs:
                return [i for i, valid in enumerate(obs[mask_key]) if valid == 1]

        # Phase-specific defaults
        phase = obs.get('phase', '')
        if phase == 'voting':
            return [0, 1]
        elif phase == 'prez_claim':
            return [0, 1, 2, 3]
        elif phase == 'chanc_claim':
            return [0, 1, 2]

        return []