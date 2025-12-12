"""Selfish agent from the ISMCTS Secret Hitler paper.

Algorithm 1: The Selfish Algorithm
- If can enact own party's policy, do it
- Else if can discard opposing party's policy, do it
- Else choose uniformly at random

Reference: "Competing in a Complex Hidden Role Game with
Information Set Monte Carlo Tree Search" (arxiv:2005.07156)
"""

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))

from agent import BaseAgent


class SelfishAgent(BaseAgent):
    """
    Selfish agent that prioritizes its own party's policies.

    For policy selection phases:
    - Liberals try to enact liberal policies / discard fascist
    - Fascists try to enact fascist policies / discard liberal

    For other phases (nomination, voting, execution):
    - Acts randomly (no strategic behavior)
    """

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action using the selfish algorithm."""
        phase = kwargs.get("phase", None)

        # Determine party affiliation from role
        # role: 0=lib, 1=fasc, 2=hitty
        role = obs.get("role", 0)
        is_liberal = (role == 0)

        valid_actions = self.get_valid_actions(obs)
        if "card_action_mask" in obs:
            return self._selfish_card_action(obs, is_liberal)
        if valid_actions:
            return random.choice(valid_actions)
        if action_space:
            return action_space.sample()

        return 0

    def _selfish_card_action(self, obs, is_liberal):
        """
        Apply selfish algorithm for card selection.

        Actions:
        - 0 = discard liberal (play fascist if only 2 cards)
        - 1 = discard fascist (play liberal if only 2 cards)

        Cards observation encodes number of liberals:
        - For president (3 cards): [1,0,0,0] = 0 libs, [0,1,0,0] = 1 lib, etc.
        - For chancellor (2 cards): [1,0,0] = 0 libs, [0,1,0] = 1 lib, [0,0,1] = 2 libs
        """
        cards = obs.get("cards", [])
        mask = obs.get("card_action_mask", [1, 1])

        # Decode number of liberal cards
        num_libs = 0
        for i, v in enumerate(cards):
            if v == 1:
                num_libs = i
                break

        # Determine number of fascist cards
        total_cards = 3 if len(cards) == 4 else 2  # prez has 3, chanc has 2
        num_fascs = total_cards - num_libs

        # Valid actions based on mask
        valid = [i for i, v in enumerate(mask) if v == 1]

        if is_liberal:
            # Liberal wants to:
            # 1. Enact liberal policy = discard fascist (action 1)
            # 2. Discard fascist policy (action 1)
            if 1 in valid and num_fascs > 0:
                return 1  # Discard fascist
            if 0 in valid and num_libs > 0:
                return 0  # Forced to discard liberal
        else:
            # Fascist wants to:
            # 1. Enact fascist policy = discard liberal (action 0)
            # 2. Discard liberal policy (action 0)
            if 0 in valid and num_libs > 0:
                return 0  # Discard liberal
            if 1 in valid and num_fascs > 0:
                return 1  # Forced to discard fascist

        # Fallback: random valid action
        return random.choice(valid) if valid else 0


class SelfishLiberalAgent(SelfishAgent):
    """Selfish agent that always plays as liberal (for testing)."""

    def get_action(self, obs, action_space=None, **kwargs):
        # Override role to always be liberal
        obs = dict(obs)
        obs["role"] = 0
        return super().get_action(obs, action_space, **kwargs)


class SelfishFascistAgent(SelfishAgent):
    """Selfish agent that always plays as fascist (for testing)."""

    def get_action(self, obs, action_space=None, **kwargs):
        # Override role to always be fascist
        obs = dict(obs)
        obs["role"] = 1
        return super().get_action(obs, action_space, **kwargs)


# For backwards compatibility
__all__ = ["SelfishAgent", "SelfishLiberalAgent", "SelfishFascistAgent"]


if __name__ == "__main__":
    # Test the selfish agent
    from eval_agent import evaluate_agents
    from agent import SimpleRandomAgent

    print("Testing SelfishAgent vs RandomAgent...")
    print()

    # All selfish agents
    print("All Selfish Agents:")
    results = evaluate_agents([SelfishAgent] * 5, num_games=1000, verbose=True, seed=42)
    print(f"  Liberal win rate: {results['lib_win_rate']:.2%}")
    print(f"  Fascist win rate: {results['fasc_win_rate']:.2%}")
    print()

    # All random agents (baseline)
    print("All Random Agents (baseline):")
    results = evaluate_agents([SimpleRandomAgent] * 5, num_games=1000, verbose=True, seed=42)
    print(f"  Liberal win rate: {results['lib_win_rate']:.2%}")
    print(f"  Fascist win rate: {results['fasc_win_rate']:.2%}")
