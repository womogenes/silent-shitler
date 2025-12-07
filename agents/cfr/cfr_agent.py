"""External Sampling MCCFR implementation for Silent Shitler."""

import copy
import random
from collections import defaultdict

from .infoset import get_infoset_key


class CFRAgent:
    """
    CFR agent using External Sampling MCCFR.
    
    Stores regret and strategy sums per information set.
    """
    
    def __init__(self):
        # infoset_key -> {action: regret_sum}
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        # infoset_key -> {action: strategy_sum}
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
    
    def get_strategy(self, infoset_key, legal_actions):
        """Get current strategy via regret matching."""
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
    
    def get_action(self, obs, action_space, phase):
        """Get action for gameplay (uses average strategy)."""
        agent_idx = obs["president_idx"]  # approximate
        infoset_key = get_infoset_key(obs, phase, agent_idx)
        legal_actions = self._get_legal_actions_from_obs(obs, action_space)
        strategy = self.get_average_strategy(infoset_key, legal_actions)
        return self.sample_action(strategy)
    
    def _get_legal_actions_from_obs(self, obs, action_space):
        """Extract legal actions from observation masks."""
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
    
    if "nomination_mask" in obs:
        return [i for i, v in enumerate(obs["nomination_mask"]) if v == 1]
    elif "execution_mask" in obs:
        return [i for i, v in enumerate(obs["execution_mask"]) if v == 1]
    elif "card_action_mask" in obs:
        return [i for i, v in enumerate(obs["card_action_mask"]) if v == 1]
    else:
        return list(range(action_space.n))


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
    phase = env.phase
    
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
    phase = env.phase
    
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
