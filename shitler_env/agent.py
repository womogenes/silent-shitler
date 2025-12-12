"""Minimal agent interface for Silent Shitler."""

import random

class BaseAgent:
    """Base agent interface for Silent Shitler agents."""
    def get_action(self, obs, action_space=None, **kwargs):
        """
        Get action given observation.

        Args:
            obs: Dictionary observation from environment
            action_space: Gymnasium action space (optional)
            **kwargs: Additional arguments (e.g., phase, deterministic)

        Returns:
            int: Action to take
        """
        pass

    def get_valid_actions(self, obs):
        """
        Extract valid actions from observation masks.

        Returns:
            list: Valid action indices, or None if no mask present
        """
        # Check for various action masks in observation
        for mask_key in ["nomination_mask", "execution_mask", "card_action_mask"]:
            if mask_key in obs:
                return [i for i, v in enumerate(obs[mask_key]) if v == 1]

        # For voting phase, valid actions are always [0, 1] (NO, YES)
        phase = obs.get("phase", "")
        if phase == "voting":
            return [0, 1]

        # For claim phases, valid actions are also [0, 1]
        if phase in ["prez_claim", "chanc_claim"]:
            return [0, 1]

        return None


class SimpleRandomAgent(BaseAgent):
    """Simple random agent using the base interface."""

    def get_action(self, obs, action_space=None, **kwargs):
        """Get random valid action."""

        # Try to get valid actions from masks
        valid_actions = self.get_valid_actions(obs)
        if valid_actions:
            return random.choice(valid_actions)

        # Fall back to sampling from action space
        if action_space:
            return action_space.sample()

        # Default to action 0 if no other option
        return 0
