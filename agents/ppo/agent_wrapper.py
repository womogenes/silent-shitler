"""PPO Agent wrapper using BaseAgent interface."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "shitler_env"))
from agent import BaseAgent

# Import internal PPO implementation
from .ppo_agent import PPOAgent as InternalPPOAgent
from .observation import ObservationProcessor

# Map phases to indices for PPO
PHASE_TO_IDX = {
    "nomination": 0,
    "voting": 1,
    "prez_cardsel": 2,
    "chanc_cardsel": 3,
    "prez_claim": 4,
    "chanc_claim": 5,
    "execution": 6,
}


class PPOAgent(BaseAgent):
    """PPO Agent wrapper conforming to BaseAgent interface."""

    def __init__(self, obs_dim=None, **kwargs):
        """Initialize PPO agent with optional parameters."""
        # If obs_dim not provided, use default from ObservationProcessor
        if obs_dim is None:
            processor = ObservationProcessor()
            obs_dim = processor.obs_dim

        self.internal_agent = InternalPPOAgent(obs_dim=obs_dim, **kwargs)
        self.obs_processor = ObservationProcessor()
        self.current_phase = None

    def _get_action_impl(self, obs, action_space=None, **kwargs):
        """
        Get action from PPO agent.

        Args:
            obs: Dictionary observation from environment
            action_space: Gymnasium action space (optional)
            **kwargs: Additional arguments (e.g., deterministic, phase)
        """
        # Get phase from observation (now included) or kwargs
        phase = kwargs.get("phase")
        if phase is None and "phase" in obs:
            phase = obs["phase"]
        elif phase is None:
            # Fallback to inference for backwards compatibility
            phase = self._infer_phase(obs, action_space)

        phase_idx = PHASE_TO_IDX.get(phase, 0)

        # Process observation to flat vector
        processed_obs = self.obs_processor.process(obs)

        # Get action mask
        action_mask = self._get_action_mask(obs, action_space)

        # Get action from internal agent (returns action, log_prob, value)
        deterministic = kwargs.get("deterministic", False)
        result = self.internal_agent.get_action(
            processed_obs, phase_idx, action_mask, deterministic
        )

        # Extract just the action (first element of tuple)
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result

        # Ensure action is a simple integer
        if hasattr(action, "item"):
            return action.item()  # Convert tensor to int
        else:
            return int(action)

    def _infer_phase(self, obs, action_space=None):
        """Infer current phase from observation and action space."""
        # Check for phase-specific observation keys
        if "nomination_mask" in obs:
            return "nomination"
        elif "execution_mask" in obs:
            return "execution"
        elif "card_action_mask" in obs:
            # Need to distinguish between prez and chanc
            if "cards" in obs and len(obs["cards"]) == 4:
                return "prez_cardsel"
            elif "cards" in obs and len(obs["cards"]) == 3:
                return "chanc_cardsel"
            # Fallback: use action mask size
            elif action_space and action_space.n == 2:
                # Could be prez_cardsel or chanc_cardsel
                return "prez_cardsel"  # Default to prez

        # Use action space size to distinguish remaining phases
        if action_space:
            if action_space.n == 4:
                return "prez_claim"
            elif action_space.n == 3:
                return "chanc_claim"
            elif action_space.n == 2:
                return "voting"
            elif action_space.n == 5:
                # Could be nomination or execution, but those should have masks
                return "nomination"

        # Default to voting
        return "voting"

    def _get_action_mask(self, obs, action_space):
        """Extract action mask from observation."""
        # Use parent class method to get valid actions
        valid_actions = self.get_valid_actions(obs)

        if valid_actions is not None:
            # Convert to mask array
            if action_space:
                mask = np.zeros(action_space.n)
                for idx in valid_actions:
                    mask[idx] = 1
                return mask
            else:
                # Assume max size from valid actions
                max_size = max(valid_actions) + 1
                mask = np.zeros(max_size)
                for idx in valid_actions:
                    mask[idx] = 1
                return mask

        # No mask found - all actions valid
        if action_space:
            return np.ones(action_space.n)
        else:
            # Default mask size
            return np.ones(5)