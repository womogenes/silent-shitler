"""DeepRole agent for playing Secret Hitler using trained neural networks."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import pickle

from shitler_env.agent import BaseAgent
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.belief import BeliefTracker
from agents.deeprole.belief_with_strategies import StrategicBeliefUpdater
from agents.deeprole.game_state import create_game_at_state
from agents.deeprole.networks import NetworkEnsemble


class DeepRoleAgent(BaseAgent):
    """DeepRole agent using CFR with neural network value functions."""

    def __init__(self, networks_path="trained_networks.pkl", cfr_iterations=50, max_depth=3):
        """Initialize DeepRole agent.

        Args:
            networks_path: Path to trained neural networks
            cfr_iterations: Number of CFR iterations for real-time planning
            max_depth: Maximum search depth for CFR
        """
        super().__init__()
        self.requires_full_state = True  # Signal that we need full state access

        # Load trained networks
        self.networks = NetworkEnsemble()
        if Path(networks_path).exists():
            self.networks.load(networks_path)
            print(f"Loaded networks from {networks_path}")
        else:
            print(f"Warning: No networks found at {networks_path}, using random play")
            self.networks = {}

        # CFR solver for real-time planning
        self.cfr = VectorCFR()
        self.cfr_iterations = cfr_iterations
        self.max_depth = max_depth

        # Belief tracker (for CFR during training)
        self.belief_tracker = BeliefTracker()

        # Strategic belief updater (full implementation with CFR strategies)
        self.strategic_belief_updater = StrategicBeliefUpdater()

        # Game state tracking
        self.player_idx = None
        self.current_belief = None

        # Store CFR strategies and environment state
        self.stored_strategies = None
        self.last_env_state = None

        # Statistics tracking
        self.action_stats = {
            'cfr_strategy': 0,  # Actions from CFR-computed strategy
            'cfr_fallback': 0,  # Random fallback when no strategy found
            'cfr_error': 0,     # Random due to CFR error
            'total': 0
        }

    def reset(self, player_idx=None):
        """Reset agent for new game.

        Args:
            player_idx: Index of the player this agent represents (0-4)
        """
        self.player_idx = player_idx
        # Initialize uniform belief over assignments (20 possible)
        self.current_belief = np.ones(20) / 20
        # Reset stored CFR strategies and environment state
        self.stored_strategies = None
        self.last_env_state = None
        # Reset CFR solver state
        self.cfr = VectorCFR()

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action using DeepRole algorithm.

        Args:
            obs: Observation dictionary from environment
            action_space: Action space (optional)

        Returns:
            Selected action
        """
        # Initialize belief if needed
        if self.current_belief is None:
            self.current_belief = np.ones(20) / 20

        # Extract player index from agent_name in kwargs if available
        if 'agent_name' in kwargs and self.player_idx is None:
            agent_name = kwargs['agent_name']
            if agent_name.startswith('P'):
                self.player_idx = int(agent_name[1])

        # Fallback: extract player index from observation if not set
        if self.player_idx is None:
            # Try to infer from observation - use agent_selection
            agent_name = obs.get('agent_selection', 'P0')
            if agent_name and agent_name.startswith('P'):
                self.player_idx = int(agent_name[1])
            else:
                self.player_idx = 0  # Default to player 0

        # Update belief based on previous CFR strategies (if available)
        if self.stored_strategies is not None:
            self._update_belief(obs)

        # Get current game state from observation
        lib_policies = obs.get("lib_policies", 0)
        fasc_policies = obs.get("fasc_policies", 0)
        president_idx = obs.get("president_idx", 0)

        # Get valid actions
        valid_actions = self.get_valid_actions(obs)
        if not valid_actions:
            return 0

        # According to DeepRole, ALL decisions should use CFR-computed strategies
        # The only exception is when we don't have networks or CFR fails
        phase = obs.get("phase", "")

        # Debug
        debug_phases = False
        if debug_phases and phase == "voting":
            print(f"\n[GET_ACTION] Phase: {phase}, Valid actions: {valid_actions}")

        # Try to use CFR for all decisions
        if phase in ["nomination", "execution", "prez_cardsel", "chanc_cardsel",
                     "voting", "prez_claim", "chanc_claim"]:
            return self._get_cfr_action(obs, valid_actions)

        # Fallback for any unknown phases
        return valid_actions[0] if valid_actions else 0

    def _update_belief(self, obs):
        """Update belief state based on observation.

        This now implements the FULL DeepRole belief update using
        strategies computed by real-time CFR (Algorithm 2).
        """
        if self.current_belief is None or self.stored_strategies is None:
            # Can't update without initial belief or strategies
            return

        # Add player index to observation for belief updater
        obs_with_idx = obs.copy()
        obs_with_idx['player_idx'] = self.player_idx

        # Update belief using CFR-computed strategies!
        # This implements: b[ρ] = b[ρ] * ∏_i π_i(I_i(h, ρ))
        self.current_belief = self.strategic_belief_updater.update_belief_with_strategies(
            self.current_belief,
            obs_with_idx,
            self.stored_strategies,
            self.last_env_state
        )

    def _get_cfr_action(self, obs, valid_actions, game_state=None):
        """Get action using CFR planning according to DeepRole Algorithm 1.

        Args:
            obs: Observation dictionary
            valid_actions: List of valid actions
            game_state: Optional full game state dict for perfect reconstruction

        Returns:
            Selected action
        """
        # If we have full game state, use it directly - much cleaner!
        if game_state is not None:
            from shitler_env.game import ShitlerEnv
            env = ShitlerEnv.from_state_dict(game_state)
        else:
            # Fall back to reconstruction from observation (legacy path)
            # This should rarely be used once the evaluation framework is updated
            env = self._reconstruct_env_from_obs(obs)

        # Get neural networks for current state
        network_key = (lib_policies, fasc_policies)
        neural_nets = None
        if hasattr(self.networks, 'networks') and network_key in self.networks.networks:
            neural_nets = {network_key: self.networks.networks[network_key]}

        # Run CFR to get strategy
        try:
            # Ensure belief is not None
            if self.current_belief is None:
                self.current_belief = np.ones(20) / 20

            # According to Algorithm 1: SOLVESITUATION returns counterfactual values
            # AND computes strategies for all players
            values = self.cfr.solve_situation(
                env,
                self.current_belief,
                num_iterations=self.cfr_iterations,
                averaging_delay=self.cfr_iterations // 3,
                neural_nets=neural_nets,
                max_depth=self.max_depth
            )

            # Extract the computed strategies from CFR
            # These represent what rational players would do
            self.stored_strategies = self.cfr.get_average_strategies()
            self.last_env_state = env

            # Debug: Check what strategies we got
            debug = False  # Enable for debugging
            if debug:
                print(f"[DEBUG] Got {len(self.stored_strategies)} strategies from CFR")
                if self.stored_strategies:
                    sample_keys = list(self.stored_strategies.keys())[:3]
                    for k in sample_keys:
                        print(f"  {k}: {self.stored_strategies[k]}")

            # Get the computed strategy for current infoset
            # The CFR solver stores strategies in strategy_sums
            # Make sure player_idx is valid
            if self.player_idx is None:
                print(f"Warning: player_idx is None, defaulting to 0")
                self.player_idx = 0

            # Override the environment's role to match our actual role
            # This is necessary because the environment was created with random roles
            # but we know our own role from the observation
            original_roles = env.roles.copy() if hasattr(env, 'roles') else {}
            if hasattr(env, 'roles') and f"P{self.player_idx}" in env.roles:
                role_map = {0: 'lib', 1: 'fasc', 2: 'hitty'}
                our_role = obs.get('role', 0)
                env.roles[f"P{self.player_idx}"] = role_map.get(our_role, 'lib')

            infoset_key = self.cfr._get_infoset_key(env, self.player_idx)

            # Restore original roles
            if hasattr(env, 'roles'):
                env.roles = original_roles

            if debug:
                print(f"[DEBUG] Player idx: {self.player_idx}")
                print(f"[DEBUG] Looking for infoset key: {infoset_key}")
                print(f"[DEBUG] Keys in stored_strategies: {list(self.stored_strategies.keys())[:5]}")

            # Use the average strategies, not the raw sums
            if infoset_key in self.stored_strategies:
                probs = self.stored_strategies[infoset_key]
                if probs:
                    # Sample action according to strategy
                    actions = list(probs.keys())
                    probabilities = [probs[a] for a in actions]
                    # Filter to valid actions only
                    valid_probs = []
                    valid_acts = []
                    for a, p in zip(actions, probabilities):
                        if a in valid_actions:
                            valid_acts.append(a)
                            valid_probs.append(p)

                    if valid_acts and sum(valid_probs) > 0:
                        # Normalize and sample
                        valid_probs = np.array(valid_probs) / sum(valid_probs)
                        action = np.random.choice(valid_acts, p=valid_probs)

                        if debug:
                            print(f"  CFR strategy found: {probs}")
                            print(f"  Valid acts: {valid_acts}, Valid probs: {valid_probs}")
                            print(f"  Selected action: {action}")

                        self.action_stats['cfr_strategy'] += 1
                        self.action_stats['total'] += 1
                        if self.action_stats['total'] % 50 == 0:  # Report every 50 actions
                            self._report_stats()
                        return action

            # Fallback: uniform random over valid actions
            if debug:
                print(f"[DEBUG] FALLBACK - no strategy found for {infoset_key}")
            self.action_stats['cfr_fallback'] += 1
            self.action_stats['total'] += 1
            if self.action_stats['total'] % 50 == 0:
                self._report_stats()
            return np.random.choice(valid_actions)

        except Exception as e:
            import traceback
            if self.action_stats['cfr_error'] == 0:  # Only print full trace on first error
                print(f"CFR failed with full trace:")
                traceback.print_exc()
            else:
                print(f"CFR failed: {e}, using random action")
            self.action_stats['cfr_error'] += 1
            self.action_stats['total'] += 1
            if self.action_stats['total'] % 50 == 0:
                self._report_stats()
            return np.random.choice(valid_actions)

    def _report_stats(self):
        """Report action selection statistics."""
        total = self.action_stats['total']
        if total > 0:
            strategy_pct = self.action_stats['cfr_strategy'] / total * 100
            fallback_pct = self.action_stats['cfr_fallback'] / total * 100
            error_pct = self.action_stats['cfr_error'] / total * 100
            print(f"DeepRole action stats (n={total}): "
                  f"CFR strategy: {strategy_pct:.1f}%, "
                  f"Fallback: {fallback_pct:.1f}%, "
                  f"Error: {error_pct:.1f}%")


class SimpleDeepRoleAgent(BaseAgent):
    """Simplified DeepRole agent for faster play."""

    def __init__(self, networks_path="trained_networks.pkl"):
        """Initialize simplified DeepRole agent."""
        super().__init__()

        # Load networks but use simpler decision making
        self.networks = NetworkEnsemble()
        if Path(networks_path).exists():
            self.networks.load(networks_path)

        self.player_idx = None

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action using simplified heuristics inspired by DeepRole."""
        phase = obs.get("phase", "")

        # Get valid actions
        valid_actions = self.get_valid_actions(obs)
        if not valid_actions:
            if phase == "voting":
                return 1  # Default vote yes
            return 0

        # Voting: simple majority heuristic
        if phase == "voting":
            # Vote yes early, no late
            total_policies = obs.get("lib_policies", 0) + obs.get("fasc_policies", 0)
            return 1 if total_policies < 5 else 0

        # Nomination: avoid recently in government
        if phase == "nomination":
            # Prefer players who haven't been in government recently
            hist_president = obs.get("hist_president", [])
            hist_chancellor = obs.get("hist_chancellor", [])
            recent = hist_president[-2:] + hist_chancellor[-2:]

            for action in valid_actions:
                if action not in recent:
                    return action
            return valid_actions[0]

        # Card selection: play fascist if liberal
        if phase in ["prez_cardsel", "chanc_cardsel"]:
            # If we have both, prefer discarding fascist
            if len(valid_actions) > 1:
                return 1  # Discard fascist
            return valid_actions[0]

        # Execution: target suspicious players (simplified)
        if phase == "execution":
            # Random for now
            return np.random.choice(valid_actions)

        # Claims: be somewhat honest
        if phase == "prez_claim":
            return 2  # Claim 2 libs (safe middle ground)
        if phase == "chanc_claim":
            return 1  # Claim 1 lib

        # Default
        return valid_actions[0] if valid_actions else 0