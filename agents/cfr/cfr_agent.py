"""CFR agent implementation for Silent Shitler."""

import sys
from pathlib import Path
import copy
import random
from collections import defaultdict

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "shitler_env"))
from agent import BaseAgent

from .infoset import get_infoset_key


class CFRAgent(BaseAgent):
    """
    CFR agent using External Sampling MCCFR.

    Inherits from BaseAgent for standard interface compatibility.
    Stores regret and strategy sums per information set.
    """

    def __init__(self):
        # infoset_key -> {action: regret_sum}
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        # infoset_key -> {action: strategy_sum}
        self.strategy_sums = defaultdict(lambda: defaultdict(float))

    def _get_action_impl(self, obs, action_space=None, **kwargs):
        """
        Get action from CFR agent using average strategy.

        Args:
            obs: Dictionary observation from environment
            action_space: Gymnasium action space (optional)
            **kwargs: Additional arguments (e.g., phase)
        """
        # Get phase from observation (now included) or kwargs
        phase = kwargs.get("phase")
        if phase is None and "phase" in obs:
            phase = obs["phase"]
        elif phase is None:
            # Fallback to inference for backwards compatibility
            phase = self._infer_phase(obs)

        # Get legal actions using parent class method
        legal_actions = self.get_valid_actions(obs)

        # If no mask found, use action space to get all actions
        if legal_actions is None and action_space:
            legal_actions = list(range(action_space.n))
        elif legal_actions is None:
            # Default to standard action space sizes based on phase
            legal_actions = self._get_default_actions(phase)

        # Get agent index (approximate from observation)
        agent_idx = self._get_agent_idx(obs)

        # Get infoset key and strategy
        infoset_key = get_infoset_key(obs, phase, agent_idx)
        strategy = self.get_average_strategy(infoset_key, legal_actions)

        # Sample action from strategy
        return self.sample_action(strategy)

    def get_strategy(self, infoset_key, legal_actions):
        """Get current strategy via regret matching (for training)."""
        regrets = self.regret_sums[infoset_key]

        # Positive regrets only
        positive_regrets = {a: max(0, regrets[a]) for a in legal_actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in legal_actions}
        else:
            # Uniform
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def get_average_strategy(self, infoset_key, legal_actions):
        """Get average strategy (converges to Nash)."""
        strat_sums = self.strategy_sums[infoset_key]
        total = sum(strat_sums[a] for a in legal_actions)

        if total > 0:
            return {a: strat_sums[a] / total for a in legal_actions}
        else:
            n = len(legal_actions)
            return {a: 1.0 / n for a in legal_actions}

    def sample_action(self, strategy):
        """Sample action from strategy distribution."""
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]
        return random.choices(actions, weights=probs, k=1)[0]

    def _infer_phase(self, obs):
        """Infer current phase from observation (fallback)."""
        # Check for phase-specific observation keys
        if "nomination_mask" in obs:
            return "nomination"
        elif "execution_mask" in obs:
            return "execution"
        elif "card_action_mask" in obs:
            # Need to distinguish between card phases
            if "cards" in obs and len(obs["cards"]) == 4:
                return "prez_cardsel"
            elif "cards" in obs and len(obs["cards"]) == 3:
                return "chanc_cardsel"
        # Could be voting or claim phases
        # Default to voting as it's most common
        return "voting"

    def _get_agent_idx(self, obs):
        """Get agent index from observation."""
        # Try to determine from president/chancellor info
        if "president_idx" in obs:
            # This is approximate - would need more info for exact agent
            return obs["president_idx"]
        return 0  # Default

    def _get_default_actions(self, phase):
        """Get default action space for phase."""
        if phase in ["nomination", "execution"]:
            return list(range(5))  # Player indices
        elif phase == "voting":
            return [0, 1]  # No/Yes
        elif phase in ["prez_cardsel", "chanc_cardsel"]:
            return [0, 1]  # Discard lib/fasc
        elif phase == "prez_claim":
            return list(range(4))  # 0L3F to 3L0F
        elif phase == "chanc_claim":
            return list(range(3))  # 0L2F to 2L0F
        else:
            return [0]  # Default single action

    # Methods for CFR training (not part of BaseAgent interface)
    def update_regrets(self, infoset_key, action_values, sample_action, weight):
        """Update regrets for CFR (for training)."""
        regrets = self.regret_sums[infoset_key]
        for action, value in action_values.items():
            regrets[action] += weight * (value - action_values[sample_action])

    def update_strategy(self, infoset_key, strategy, weight):
        """Update strategy sums for CFR (for training)."""
        strat_sums = self.strategy_sums[infoset_key]
        for action, prob in strategy.items():
            strat_sums[action] += weight * prob


# Backward compatibility alias
class InternalCFRAgent(CFRAgent):
    """Alias for backward compatibility with training scripts."""
    pass


def get_legal_actions_helper(obs, action_space):
    """Helper to extract legal actions from observation."""
    if "nomination_mask" in obs:
        return [i for i, v in enumerate(obs["nomination_mask"]) if v == 1]
    elif "execution_mask" in obs:
        return [i for i, v in enumerate(obs["execution_mask"]) if v == 1]
    elif "card_action_mask" in obs:
        return [i for i, v in enumerate(obs["card_action_mask"]) if v == 1]
    else:
        return list(range(action_space.n))


def get_legal_actions(env, agent):
    """Get legal actions for agent in current state."""
    obs = env.observe(agent)
    action_space = env.action_space(agent)

    if obs is None:
        return []

    return get_legal_actions_helper(obs, action_space)


def outcome_sampling_cfr(env, cfr_agent, traverser_idx, pi_i=1.0, pi_neg_i=1.0, pi_sample=1.0):
    """
    Outcome Sampling MCCFR - samples a single trajectory.

    Much faster than external sampling for deep games.
    Updates regrets along the sampled path using importance sampling.

    Args:
        env: Game environment (will be mutated)
        cfr_agent: CFR agent with regret/strategy tables
        traverser_idx: Index of player we're computing regrets for
        pi_i: Reach probability for traverser
        pi_neg_i: Reach probability for opponents
        pi_sample: Sampling probability for this trajectory

    Returns:
        (utility, tail_prob) tuple
    """
    if all(env.terminations.values()):
        traverser_agent = env.agents[traverser_idx]
        return env.rewards[traverser_agent], 1.0

    current_agent = env.agent_selection
    current_idx = env.agents.index(current_agent)
    obs = env.observe(current_agent)
    phase = obs.get("phase", env.phase)  # Use phase from observation if available

    legal_actions = get_legal_actions(env, current_agent)
    if not legal_actions:
        return 0.0, 1.0

    infoset_key = get_infoset_key(obs, phase, current_idx)
    strategy = cfr_agent.get_strategy(infoset_key, legal_actions)

    # Sample action according to strategy
    action = cfr_agent.sample_action(strategy)
    action_prob = strategy[action]

    # Take action and recurse
    env.step(action)

    if current_idx == traverser_idx:
        utility, tail = outcome_sampling_cfr(
            env, cfr_agent, traverser_idx,
            pi_i * action_prob, pi_neg_i, pi_sample * action_prob
        )

        # Importance-sampled counterfactual value
        W = utility * pi_neg_i / pi_sample

        # Update regrets for all actions at this infoset
        for a in legal_actions:
            if a == action:
                regret = W * (1 - action_prob)
            else:
                regret = -W * action_prob
            cfr_agent.regret_sums[infoset_key][a] += regret

        # Update strategy sum
        for a in legal_actions:
            cfr_agent.strategy_sums[infoset_key][a] += pi_i * strategy[a] / pi_sample

        return utility, tail * action_prob
    else:
        utility, tail = outcome_sampling_cfr(
            env, cfr_agent, traverser_idx,
            pi_i, pi_neg_i * action_prob, pi_sample * action_prob
        )
        return utility, tail * action_prob


# Keep external sampling for reference but rename
def external_sampling_cfr(env, cfr_agent, traverser_idx, reach_prob=1.0):
    """
    External Sampling MCCFR - explores all traverser actions.

    WARNING: Very slow for deep games like Secret Hitler.
    Use outcome_sampling_cfr instead.
    """
    if all(env.terminations.values()):
        traverser_agent = env.agents[traverser_idx]
        return env.rewards[traverser_agent]

    current_agent = env.agent_selection
    current_idx = env.agents.index(current_agent)
    obs = env.observe(current_agent)
    phase = obs.get("phase", env.phase)  # Use phase from observation if available

    legal_actions = get_legal_actions(env, current_agent)
    if not legal_actions:
        return 0.0

    infoset_key = get_infoset_key(obs, phase, current_idx)
    strategy = cfr_agent.get_strategy(infoset_key, legal_actions)

    if current_idx == traverser_idx:
        action_values = {}
        for action in legal_actions:
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            action_values[action] = external_sampling_cfr(
                env_copy, cfr_agent, traverser_idx, reach_prob
            )

        ev = sum(strategy[a] * action_values[a] for a in legal_actions)

        for action in legal_actions:
            regret = action_values[action] - ev
            cfr_agent.regret_sums[infoset_key][action] += regret

        for action in legal_actions:
            cfr_agent.strategy_sums[infoset_key][action] += reach_prob * strategy[action]

        return ev
    else:
        action = cfr_agent.sample_action(strategy)
        env.step(action)
        return external_sampling_cfr(env, cfr_agent, traverser_idx, reach_prob * strategy[action])