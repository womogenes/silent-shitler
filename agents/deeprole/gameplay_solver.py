"""Depth-limited solver for DeepRole gameplay with neural network evaluation."""

import numpy as np
import torch
import copy
from pathlib import Path

from .role_assignments import RoleAssignmentManager
from .belief import BeliefTracker
from .vector_cfr import VectorCFR
from .networks import NetworkEnsemble


class DeepRoleSolver:
    """Depth-limited solver that uses neural networks for leaf evaluation.

    This is the main component used during actual gameplay to select actions.
    It runs short CFR searches with neural network evaluation at leaves.
    """

    def __init__(self, network_path=None):
        """Initialize solver with optional pre-trained networks.

        Args:
            network_path: Path to saved neural networks
        """
        self.manager = RoleAssignmentManager()
        self.belief_tracker = BeliefTracker()
        self.cfr = VectorCFR()
        self.networks = NetworkEnsemble()

        if network_path and Path(network_path).exists():
            self.networks.load(network_path)
            print(f"Loaded {len(self.networks.networks)} networks from {network_path}")

    def get_action(self, env, player_idx, belief=None, cfr_iterations=50,
                   cfr_depth=3, temperature=1.0):
        """Get action for player using depth-limited CFR with neural evaluation.

        Args:
            env: Current game environment
            player_idx: Index of acting player
            belief: Current belief distribution (if None, uses uniform)
            cfr_iterations: Number of CFR iterations to run
            cfr_depth: Maximum search depth before using neural evaluation
            temperature: Temperature for action sampling (1.0 = proportional, 0 = greedy)

        Returns:
            Selected action
        """
        if belief is None:
            belief = self.manager.get_uniform_belief()

        # Get current game state
        lib_policies = env.lib_policies
        fasc_policies = env.fasc_policies

        # Check if we have a network for this state
        has_network = (lib_policies, fasc_policies) in self.networks.networks

        if not has_network and (lib_policies + fasc_policies) > 3:
            # For later game states without networks, use simpler heuristics
            return self._get_heuristic_action(env, player_idx, belief)

        # Run depth-limited CFR
        env_copy = copy.deepcopy(env)

        # Create modified CFR that uses neural networks at leaves
        cfr_with_nn = DepthLimitedCFR(self.networks.networks, max_depth=cfr_depth)

        # Solve for a limited number of iterations
        cfr_with_nn.solve_from_position(
            env_copy, belief, player_idx,
            num_iterations=cfr_iterations
        )

        # Extract strategy for current player's information set
        strategy = cfr_with_nn.get_average_strategy(env, player_idx)

        if not strategy:
            # Fallback to random legal action
            return self._get_random_action(env, player_idx)

        # Sample action based on strategy and temperature
        return self._sample_action(strategy, temperature)

    def _get_heuristic_action(self, env, player_idx, belief):
        """Simple heuristic for when neural networks aren't available."""
        agent = env.agents[player_idx]
        obs = env.observe(agent)
        role = obs['role']

        # Get legal actions
        legal_actions = []
        for mask_key in ['nomination_mask', 'execution_mask', 'card_action_mask']:
            if mask_key in obs:
                legal_actions = [i for i, valid in enumerate(obs[mask_key]) if valid == 1]
                if legal_actions:
                    break

        if not legal_actions:
            return 0

        # Simple role-based heuristics
        if env.phase == 'nomination':
            # Nominate someone likely to be same team based on belief
            expected_roles = self._compute_expected_roles(belief)

            # Find best match for same team
            best_action = legal_actions[0]
            best_score = -1

            for action in legal_actions:
                nominee = action
                if role == 0:  # Liberal
                    score = 1 - expected_roles[nominee]  # Lower expected role = more likely liberal
                else:  # Fascist/Hitler
                    score = expected_roles[nominee]  # Higher expected role = more likely fascist

                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action

        elif env.phase == 'voting':
            # Vote based on expected alignment of government
            president = env.president_idx
            chancellor = env.chancellor_nominee

            expected_roles = self._compute_expected_roles(belief)
            gov_fascist_prob = (expected_roles[president] + expected_roles[chancellor]) / 2

            if role == 0:  # Liberal
                return 1 if gov_fascist_prob < 0.4 else 0  # Ja if likely liberal
            else:  # Fascist/Hitler
                return 1 if gov_fascist_prob > 0.3 else 0  # Ja if likely fascist

        # Default to random
        return np.random.choice(legal_actions)

    def _compute_expected_roles(self, belief):
        """Compute expected role value for each player.

        Returns array where 0 = definitely liberal, 1 = definitely fascist/hitler
        """
        expected = np.zeros(5)
        for i, assignment in enumerate(self.manager.assignments):
            for player in range(5):
                if assignment[player] > 0:  # Fascist or Hitler
                    expected[player] += belief[i]
        return expected

    def _get_random_action(self, env, player_idx):
        """Get random legal action."""
        agent = env.agents[player_idx]
        obs = env.observe(agent)

        for mask_key in ['nomination_mask', 'execution_mask', 'card_action_mask']:
            if mask_key in obs:
                legal_actions = [i for i, valid in enumerate(obs[mask_key]) if valid == 1]
                if legal_actions:
                    return np.random.choice(legal_actions)

        return 0

    def _sample_action(self, strategy, temperature):
        """Sample action from strategy with temperature."""
        if temperature == 0:
            # Greedy selection
            return max(strategy, key=strategy.get)

        # Convert strategy to probabilities with temperature
        actions = list(strategy.keys())
        probs = np.array([strategy[a] for a in actions])

        if temperature != 1.0:
            # Apply temperature
            log_probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= probs.sum()

        return np.random.choice(actions, p=probs)

    def update_belief(self, belief, observation):
        """Update belief based on observation.

        Args:
            belief: Current belief distribution
            observation: Game observation (action taken, outcome, etc.)

        Returns:
            Updated belief distribution
        """
        return self.belief_tracker.update_belief_from_observation(belief, observation)


class DepthLimitedCFR(VectorCFR):
    """Modified CFR that uses neural network evaluation at maximum depth."""

    def __init__(self, neural_nets, max_depth=3):
        super().__init__()
        self.neural_nets = neural_nets
        self.max_depth = max_depth

    def solve_from_position(self, env, belief, player_idx, num_iterations=50):
        """Solve game from current position for specific player.

        Args:
            env: Current game environment
            belief: Belief distribution
            player_idx: Acting player index
            num_iterations: Number of CFR iterations

        Returns:
            None (strategies stored internally)
        """
        # Clear previous strategies
        self.regret_sums.clear()
        self.strategy_sums.clear()

        # Run CFR iterations with neural net evaluation
        for iteration in range(num_iterations):
            weight = max(iteration - 10, 0)  # Shorter averaging delay

            reach_probs = np.ones((self.num_players, self.manager.num_assignments))

            self._cfr_iteration(
                copy.deepcopy(env), belief, reach_probs, weight,
                self.neural_nets, depth=0, max_depth=self.max_depth
            )

    def get_average_strategy(self, env, player_idx):
        """Get average strategy for player at current state.

        Returns:
            Dict mapping actions to probabilities
        """
        infoset_key = self._get_infoset_key(env, player_idx)

        if infoset_key not in self.strategy_sums:
            return {}

        strategy_sum = self.strategy_sums[infoset_key]
        total = sum(strategy_sum.values())

        if total > 0:
            return {a: s/total for a, s in strategy_sum.items()}
        else:
            # No strategy computed, return uniform over legal actions
            legal_actions = self._get_legal_actions(env, player_idx)
            if legal_actions:
                n = len(legal_actions)
                return {a: 1.0/n for a in legal_actions}
            return {}

    def _neural_values(self, env, belief, reach_probs, neural_nets):
        """Override to actually use neural networks for evaluation.

        Args:
            env: Game environment at evaluation point
            belief: Current belief distribution
            reach_probs: Reach probabilities
            neural_nets: Dictionary of trained networks

        Returns:
            Value vector for each player
        """
        lib_policies = env.lib_policies
        fasc_policies = env.fasc_policies

        # Check if we have a network for this state
        if (lib_policies, fasc_policies) not in neural_nets:
            # No network available, use simple heuristic
            return self._heuristic_values(env, belief)

        network = neural_nets[(lib_policies, fasc_policies)]

        # Get current president
        president_idx = env.president_idx

        # Prepare input for network
        with torch.no_grad():
            # Create one-hot encoding for president
            president_tensor = torch.tensor(president_idx)
            belief_tensor = torch.tensor(belief, dtype=torch.float32)

            # Get value matrix from network
            values_matrix = network(president_tensor, belief_tensor)  # Shape (5, 20)

            # Convert to expected values for each player
            player_values = np.zeros(self.num_players)
            for player in range(self.num_players):
                # Weight values by belief and reach probabilities
                weighted_values = values_matrix[player].numpy() * belief

                # Consider reach probabilities
                reach_weight = reach_probs[player].sum()
                if reach_weight > 0:
                    player_values[player] = weighted_values.sum() / reach_weight
                else:
                    player_values[player] = weighted_values.sum()

        return player_values

    def _heuristic_values(self, env, belief):
        """Simple heuristic evaluation when no network available.

        Estimates values based on policy counts and expected roles.
        """
        lib_policies = env.lib_policies
        fasc_policies = env.fasc_policies

        # Compute expected roles
        expected_fascist = np.zeros(5)
        for i, assignment in enumerate(self.manager.assignments):
            for player in range(5):
                if assignment[player] > 0:
                    expected_fascist[player] += belief[i]

        # Simple value heuristic based on game progress
        lib_progress = lib_policies / 5.0
        fasc_progress = fasc_policies / 6.0

        values = np.zeros(5)
        for player in range(5):
            if expected_fascist[player] < 0.5:
                # Likely liberal
                values[player] = lib_progress - fasc_progress
            else:
                # Likely fascist
                values[player] = fasc_progress - lib_progress

        return values