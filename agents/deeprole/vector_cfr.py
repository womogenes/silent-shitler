"""Vector-form CFR+ implementation for Secret Hitler."""

import numpy as np
import copy
from collections import defaultdict
from .role_assignments import RoleAssignmentManager
from .belief import BeliefTracker


class VectorCFR:
    """Vector-form CFR+ with belief tracking."""

    def __init__(self, num_players=5):
        self.num_players = num_players
        self.manager = RoleAssignmentManager()
        self.belief_tracker = BeliefTracker()

        # Regret and strategy sums indexed by (role, state_features) -> action -> value
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        self.strategy_sums = defaultdict(lambda: defaultdict(float))

    def solve_situation(self, env, initial_belief, num_iterations=1500, averaging_delay=500,
                       neural_nets=None, max_depth=10):
        """Solve game situation using depth-limited CFR (Algorithm 1).

        Args:
            env: Game environment at starting position
            initial_belief: Initial belief vector over role assignments (20,)
            num_iterations: Number of CFR iterations
            averaging_delay: Iterations before starting averaging
            neural_nets: Dict of neural networks for leaf evaluation
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            Value vector for each player (5,)
        """
        cumulative_values = np.zeros(self.num_players)
        total_weight = 0

        for iteration in range(num_iterations):
            weight = max(iteration - averaging_delay, 0)
            total_weight += weight

            # Initialize reach probabilities for all players
            reach_probs = np.ones((self.num_players, self.manager.num_assignments))

            # Run modified CFR+ with depth limit
            values = self._cfr_iteration(
                copy.deepcopy(env), initial_belief, reach_probs, weight, neural_nets, depth=0, max_depth=max_depth
            )
            cumulative_values += values

        return cumulative_values / max(total_weight, 1)

    def _cfr_iteration(self, env, belief, reach_probs, weight, neural_nets, depth=0, max_depth=10):
        """Single iteration of modified CFR+ (Algorithm 1, procedure MODIFIEDCFR+).

        Returns values of shape (num_players,) not (num_players, num_assignments).
        """

        # Terminal node
        if all(env.terminations.values()):
            return self._terminal_values(env, belief, reach_probs)

        # Depth limit - return heuristic evaluation
        if depth >= max_depth:
            # Simple heuristic: uniform random values
            return np.random.randn(self.num_players) * 0.1

        # Check if we should use neural net evaluation
        if neural_nets and self._should_use_neural(env, neural_nets):
            return self._neural_values(env, belief, reach_probs, neural_nets)

        # Get acting players
        acting_players = self._get_acting_players(env)

        # Get strategies for all acting players
        strategies = {}
        infoset_keys = {}
        for player_idx in acting_players:
            infoset_key = self._get_infoset_key(env, player_idx)
            infoset_keys[player_idx] = infoset_key
            strategies[player_idx] = self._get_strategy(infoset_key, env, player_idx)

        # Handle simultaneous actions (voting)
        if len(acting_players) > 1:
            return self._handle_simultaneous(
                env, belief, reach_probs, weight, neural_nets,
                acting_players, strategies, infoset_keys, depth, max_depth
            )
        else:
            # Single player acting
            player_idx = acting_players[0]
            return self._handle_sequential(
                env, belief, reach_probs, weight, neural_nets,
                player_idx, strategies[player_idx], infoset_keys[player_idx], depth, max_depth
            )

    def _handle_simultaneous(self, env, belief, reach_probs, weight, neural_nets,
                            acting_players, strategies, infoset_keys, depth, max_depth):
        """Handle simultaneous actions (voting phase)."""
        # Enumerate all possible vote combinations
        vote_outcomes = self._enumerate_vote_outcomes(acting_players, env)

        values = np.zeros(self.num_players)  # Shape (5,)
        action_values = defaultdict(lambda: defaultdict(float))

        for outcome in vote_outcomes:
            # Probability of this outcome
            outcome_prob = 1.0
            new_reach = reach_probs.copy()

            for player_idx in acting_players:
                action = outcome[player_idx]
                strategy = strategies[player_idx]
                if action in strategy:
                    prob = strategy[action]
                    outcome_prob *= prob
                    # Update reach probabilities
                    for role in range(3):
                        role_indices = self.manager.get_infoset_indices(player_idx, role)
                        new_reach[player_idx, role_indices] *= prob

            if outcome_prob > 0:
                # Take action in environment
                env_copy = copy.deepcopy(env)
                self._apply_votes(env_copy, outcome)

                # Update belief based on outcome
                new_belief = self.belief_tracker.update_belief(
                    belief, self._get_history(env), outcome, strategies
                )

                # Recurse
                child_values = self._cfr_iteration(
                    env_copy, new_belief, new_reach, weight, neural_nets, depth + 1, max_depth
                )

                # Accumulate values
                for player_idx in range(self.num_players):
                    if player_idx in acting_players:
                        action = outcome[player_idx]
                        action_values[player_idx][action] += child_values[player_idx] * outcome_prob
                    values[player_idx] += child_values[player_idx] * outcome_prob

        # Update regrets and strategies for all acting players
        for player_idx in acting_players:
            # Convert to per-assignment values for regret updates
            action_vals_per_assignment = {}
            for action in strategies[player_idx]:
                action_vals_per_assignment[action] = action_values[player_idx].get(action, 0.0)

            self._update_regrets_strategies(
                infoset_keys[player_idx], action_vals_per_assignment,
                strategies[player_idx], reach_probs[player_idx], weight
            )

        return values

    def _handle_sequential(self, env, belief, reach_probs, weight, neural_nets,
                          player_idx, strategy, infoset_key, depth, max_depth):
        """Handle single player action."""
        values = np.zeros(self.num_players)  # Shape (5,)
        action_values = defaultdict(float)

        # Get current legal actions to ensure we don't try illegal moves
        legal_actions = self._get_legal_actions(env, player_idx)

        # Handle case with no legal actions
        if not legal_actions or not strategy:
            # Return heuristic evaluation if stuck
            return np.random.randn(self.num_players) * 0.1

        for action, prob in strategy.items():
            if prob > 0 and action in legal_actions:  # Only try legal actions
                # Take action
                env_copy = copy.deepcopy(env)
                env_copy.step(action)

                # Update reach probabilities
                new_reach = reach_probs.copy()
                for role in range(3):
                    role_indices = self.manager.get_infoset_indices(player_idx, role)
                    new_reach[player_idx, role_indices] *= prob

                # Update belief
                observation = {'action': action, 'player': player_idx}
                new_belief = self.belief_tracker.update_belief(
                    belief, self._get_history(env), observation, {player_idx: strategy}
                )

                # Recurse
                child_values = self._cfr_iteration(
                    env_copy, new_belief, new_reach, weight, neural_nets, depth + 1, max_depth
                )

                action_values[action] = child_values[player_idx]
                values += prob * child_values

        # Update regrets and strategy only if we have actions
        if action_values:
            self._update_regrets_strategies(
                infoset_key, action_values, strategy, reach_probs[player_idx], weight
            )

        return values

    def _get_strategy(self, infoset_key, env, player_idx):
        """Get current strategy via regret matching+."""
        legal_actions = self._get_legal_actions(env, player_idx)

        # If no legal actions, return empty strategy
        if not legal_actions:
            return {}

        regrets = self.regret_sums[infoset_key]

        positive_regrets = {a: max(0, regrets[a]) for a in legal_actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in legal_actions}
        else:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def _update_regrets_strategies(self, infoset_key, action_values, strategy, reach_prob, weight):
        """Update regrets and strategy sums.

        Args:
            infoset_key: Information set identifier
            action_values: Dict mapping actions to values (scalars)
            strategy: Current strategy for this infoset
            reach_prob: Reach probability vector (shape num_assignments)
            weight: Weight for strategy sum update
        """
        # Compute expected value
        ev = sum(strategy[a] * action_values[a] for a in strategy)

        # Update regrets (CFR+)
        reach_weight = np.sum(reach_prob)  # Sum over all assignments
        for action in strategy:
            regret = action_values[action] - ev
            self.regret_sums[infoset_key][action] = max(
                self.regret_sums[infoset_key][action] + regret * reach_weight, 0
            )

        # Update strategy sums for averaging
        if weight > 0:
            for action in strategy:
                self.strategy_sums[infoset_key][action] += weight * strategy[action] * reach_weight

    def _get_infoset_key(self, env, player_idx):
        """Get information set key for player."""
        obs = env.observe(env.agents[player_idx])
        role = obs['role']

        # Include relevant game state
        key = (
            role,
            obs['lib_policies'],
            obs['fasc_policies'],
            obs['election_tracker'],
            env.phase,
            tuple(obs['executed']),
        )

        # Add role-specific information
        if role in [1, 2]:  # Fascist or Hitler
            key += (tuple(obs['all_roles']),)

        return key

    def _get_acting_players(self, env):
        """Get indices of players currently acting."""
        if env.phase == 'voting':
            # All alive players vote simultaneously
            return [i for i, agent in enumerate(env.agents) if agent not in env.executed]
        else:
            # Single player acts
            return [env.agents.index(env.agent_selection)]

    def _enumerate_vote_outcomes(self, acting_players, env):
        """Enumerate all possible vote combinations."""
        from itertools import product
        outcomes = []
        for votes in product([0, 1], repeat=len(acting_players)):
            outcome = {}
            for i, player_idx in enumerate(acting_players):
                outcome[player_idx] = votes[i]
            outcomes.append(outcome)
        return outcomes

    def _apply_votes(self, env, votes):
        """Apply simultaneous votes to environment."""
        # Apply all votes
        for player_idx, vote in votes.items():
            agent = env.agents[player_idx]
            if agent not in env.executed:
                env.votes[agent] = vote

        # Check if all alive players have voted
        alive_agents = [a for a in env.agents if a not in env.executed]
        if len(env.votes) == len(alive_agents):
            # Record and process the vote
            env._record_government_attempt()
            yes_votes = sum(env.votes.values())
            if yes_votes > len(env.votes) // 2:
                env._government_succeeds()
            else:
                env._government_fails()

    def _get_legal_actions(self, env, player_idx):
        """Get legal actions for player."""
        agent = env.agents[player_idx]
        obs = env.observe(agent)
        action_space = env.action_space(agent)

        # Extract from masks if available
        for mask_key in ['nomination_mask', 'execution_mask', 'card_action_mask']:
            if mask_key in obs:
                legal = [i for i, valid in enumerate(obs[mask_key]) if valid == 1]
                # Verify we found legal actions
                if legal:
                    return legal

        # If no mask found or no legal actions, return default
        # This shouldn't happen if the environment is working correctly
        return list(range(action_space.n))

    def _get_history(self, env):
        """Extract history dict from environment."""
        return {
            'president_idx': env.president_idx,
            'chancellor_idx': env.chancellor_nominee,
            'fasc_policies': env.fasc_policies,
            'lib_policies': env.lib_policies,
        }

    def _terminal_values(self, env, belief, reach_probs):
        """Compute terminal values (Algorithm 2)."""
        values = np.zeros(self.num_players)

        # Compute terminal belief
        terminal_belief = belief * np.prod(reach_probs, axis=0)
        terminal_belief /= np.sum(terminal_belief)

        # Sum over assignments weighted by belief
        for i, assignment in enumerate(self.manager.assignments):
            for player_idx in range(self.num_players):
                role = assignment[player_idx]
                reward = env.rewards[env.agents[player_idx]]
                values[player_idx] += terminal_belief[i] * reward

        # Convert to counterfactual values
        for player_idx in range(self.num_players):
            if np.sum(reach_probs[player_idx]) > 0:
                values[player_idx] /= np.sum(reach_probs[player_idx])

        return values

    def _should_use_neural(self, env, neural_nets):
        """Check if we should use neural network evaluation."""
        # Use neural nets for depth limiting based on policies enacted
        total_policies = env.lib_policies + env.fasc_policies
        return total_policies in neural_nets

    def _neural_values(self, env, belief, reach_probs, neural_nets):
        """Get values from neural network."""
        # To be implemented with neural network integration
        raise NotImplementedError("Neural evaluation not yet implemented")