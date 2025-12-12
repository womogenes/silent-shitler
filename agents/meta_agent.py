"""Meta Agent - Smart Liberal with Suspicion Tracking.

Implements strategies from high-ELO Secret Hitler play:
- Suspicion meter for each player (softmax weighted)
- Updates suspicion based on fascist government outcomes
- Detects claim conflicts and assigns suspicion
- Always claims consistently as a liberal
- Votes for least suspicious players
- Executes most suspicious player
"""

import random
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))

from agent import BaseAgent


class MetaAgent(BaseAgent):
    """
    Smart liberal agent using meta-game suspicion tracking.

    Suspicion scoring:
    - All players start with suspicion = 1.0
    - President of fascist government: +2.0 suspicion
    - Chancellor of fascist government: +1.0 suspicion
    - Claim conflicts: both players in conflict get +1.5 suspicion
    - Uses softmax to convert suspicion to voting probabilities

    Behavior:
    - Nomination: Nominate least suspicious eligible player
    - Voting: Vote yes if avg suspicion of govt < threshold, weighted by softmax
    - Card selection: Always play liberal (discard fascist)
    - Claims: Always claim truthfully (as a liberal would)
    - Execution: Execute most suspicious player
    """

    def __init__(self, temperature=1.0):
        """
        Initialize the meta agent.

        Args:
            temperature: Softmax temperature for voting decisions.
                         Lower = more deterministic, higher = more random.
        """
        self.temperature = temperature
        self.reset_suspicion()

    def reset_suspicion(self):
        """Reset suspicion scores for a new game."""
        self.suspicion = [1.0] * 5  # All players start at 1.0
        self.last_processed_govt = -1  # Track which governments we've processed
        self.detected_conflicts = set()  # Track (prez, chanc) pairs with conflicts

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action using meta-game strategies."""
        phase = kwargs.get("phase", obs.get("phase", None))
        agent_idx = kwargs.get("agent_idx", None)

        # Update suspicion based on game history
        self._update_suspicion(obs)

        # Phase-specific behavior
        if phase == "nomination" or "nomination_mask" in obs:
            return self._handle_nomination(obs)

        elif phase == "voting":
            return self._handle_voting(obs)

        elif phase in ["prez_cardsel", "chanc_cardsel"] or "card_action_mask" in obs:
            return self._handle_card_selection(obs)

        elif phase == "prez_claim":
            return self._handle_prez_claim(obs)

        elif phase == "chanc_claim":
            return self._handle_chanc_claim(obs)

        elif phase == "execution" or "execution_mask" in obs:
            return self._handle_execution(obs)

        # Fallback
        valid_actions = self.get_valid_actions(obs)
        if valid_actions:
            return random.choice(valid_actions)
        if action_space:
            return action_space.sample()
        return 0

    def _update_suspicion(self, obs):
        """Update suspicion scores based on game history."""
        hist_succeeded = obs.get("hist_succeeded", [])
        hist_policy = obs.get("hist_policy", [])
        hist_president = obs.get("hist_president", [])
        hist_chancellor = obs.get("hist_chancellor", [])
        hist_prez_claim = obs.get("hist_prez_claim", [])
        hist_chanc_claim = obs.get("hist_chanc_claim", [])

        # Process new governments since last update
        for i in range(self.last_processed_govt + 1, len(hist_succeeded)):
            if hist_succeeded[i] != 1:
                continue  # Skip failed governments

            prez = hist_president[i]
            chanc = hist_chancellor[i]
            policy = hist_policy[i]
            prez_claim = hist_prez_claim[i]
            chanc_claim = hist_chanc_claim[i]

            # Fascist policy played - increase suspicion
            if policy == 1:  # Fascist policy
                self.suspicion[prez] += 2.0  # President more responsible
                self.suspicion[chanc] += 1.0  # Chancellor less responsible

            # Check for claim conflicts
            if prez_claim >= 0 and chanc_claim >= 0:
                conflict = self._detect_conflict(prez_claim, chanc_claim, policy)
                if conflict and (prez, chanc) not in self.detected_conflicts:
                    self.detected_conflicts.add((prez, chanc))
                    # Both players in conflict get suspicion
                    self.suspicion[prez] += 1.5
                    self.suspicion[chanc] += 1.5

        self.last_processed_govt = len(hist_succeeded) - 1

    def _detect_conflict(self, prez_claim, chanc_claim, policy_played):
        """
        Detect if there's a conflict between president and chancellor claims.

        A conflict exists if the claims are mathematically impossible.

        Args:
            prez_claim: Number of liberals president claims to have drawn (0-3)
            chanc_claim: Number of liberals chancellor claims to have received (0-2)
            policy_played: 0 = liberal played, 1 = fascist played

        Returns:
            bool: True if there's a conflict
        """
        # President draws 3, gives 2 to chancellor
        # If prez claims X liberals, they should give at most X liberals to chanc
        # (they discard 1 card)

        # Maximum libs chanc could receive = min(prez_claim, 2)
        max_libs_to_chanc = min(prez_claim, 2)

        # If chanc claims more libs than possible, conflict
        if chanc_claim > max_libs_to_chanc:
            return True

        # Check if claimed libs match what was played
        # If policy_played is liberal (0), chanc must have had at least 1 lib
        if policy_played == 0 and chanc_claim == 0:
            return True  # Played liberal but claimed 0 liberals

        # If policy_played is fascist (1) and chanc claims 2 libs, suspicious
        # (could be forced, but worth noting)

        return False

    def _softmax(self, values, temperature=None):
        """Apply softmax to convert values to probabilities."""
        if temperature is None:
            temperature = self.temperature

        # Negate suspicion (lower suspicion = higher probability)
        neg_values = [-v / temperature for v in values]

        # Numerical stability
        max_val = max(neg_values)
        exp_values = [math.exp(v - max_val) for v in neg_values]
        total = sum(exp_values)

        if total == 0:
            return [1.0 / len(values)] * len(values)

        return [v / total for v in exp_values]

    def _handle_nomination(self, obs):
        """Nominate the least suspicious eligible player."""
        mask = obs.get("nomination_mask", [])
        valid = [i for i, v in enumerate(mask) if v == 1]

        if not valid:
            return 0

        # Get suspicion scores for valid nominees
        scores = [(i, self.suspicion[i]) for i in valid]

        # Sort by suspicion (ascending - least suspicious first)
        scores.sort(key=lambda x: x[1])

        # Return least suspicious
        return scores[0][0]

    def _handle_voting(self, obs):
        """Vote based on suspicion of proposed government."""
        prez_idx = obs.get("president_idx", 0)
        chanc_idx = obs.get("chancellor_nominee", -1)

        if chanc_idx < 0:
            return 1  # Default yes if something's wrong

        # Calculate average suspicion of the government
        govt_suspicion = (self.suspicion[prez_idx] + self.suspicion[chanc_idx]) / 2

        # Compare to average suspicion of all alive players
        executed = obs.get("executed", [0] * 5)
        alive_suspicion = [
            self.suspicion[i] for i in range(5) if executed[i] == 0
        ]
        avg_suspicion = sum(alive_suspicion) / len(alive_suspicion) if alive_suspicion else 1.0

        # Vote yes if government is less suspicious than average
        # With some randomness based on how much less/more suspicious
        threshold = avg_suspicion * 1.2  # Slightly higher threshold

        if govt_suspicion < threshold:
            return 1  # Yes
        else:
            # Still might vote yes with some probability
            # Lower probability for more suspicious governments
            diff = govt_suspicion - threshold
            prob_yes = max(0.1, 1.0 / (1.0 + diff))
            return 1 if random.random() < prob_yes else 0

    def _handle_card_selection(self, obs):
        """
        Always play liberal policy (discard fascist) as a liberal would.

        Action mapping:
        - 0 = discard liberal (play fascist if only 2 cards)
        - 1 = discard fascist (play liberal if only 2 cards)
        """
        mask = obs.get("card_action_mask", [1, 1])
        cards = obs.get("cards", [])

        # Decode number of liberals
        num_libs = 0
        for i, v in enumerate(cards):
            if v == 1:
                num_libs = i
                break

        total_cards = 3 if len(cards) == 4 else 2
        num_fascs = total_cards - num_libs

        valid = [i for i, v in enumerate(mask) if v == 1]

        # As a liberal: always try to discard fascist (play liberal)
        if 1 in valid and num_fascs > 0:
            return 1  # Discard fascist
        if 0 in valid and num_libs > 0:
            return 0  # Forced to discard liberal

        return random.choice(valid) if valid else 0

    def _handle_prez_claim(self, obs):
        """
        Claim truthfully as a liberal.

        Returns number of liberals drawn (0-3).
        """
        # Get what we actually saw from personal history
        personal_cards = obs.get("personal_cards_seen", [])

        if personal_cards:
            # Most recent entry is what we just saw
            last_entry = personal_cards[-1]
            if isinstance(last_entry, (list, tuple)) and len(last_entry) >= 2:
                num_libs = last_entry[1]  # (gov_idx, num_libs, num_fascs)
                return num_libs

        # Fallback: claim based on expected distribution
        # This shouldn't happen in normal play
        return 1  # Claim 1 liberal (most common)

    def _handle_chanc_claim(self, obs):
        """
        Claim truthfully as a liberal.

        Returns number of liberals received (0-2).
        """
        personal_cards = obs.get("personal_cards_seen", [])

        if personal_cards:
            last_entry = personal_cards[-1]
            if isinstance(last_entry, (list, tuple)) and len(last_entry) >= 2:
                num_libs = last_entry[1]
                return min(num_libs, 2)  # Chancellor receives at most 2

        return 1  # Fallback claim

    def _handle_execution(self, obs):
        """Execute the most suspicious player."""
        mask = obs.get("execution_mask", [])
        valid = [i for i, v in enumerate(mask) if v == 1]

        if not valid:
            return 0

        # Get suspicion scores for valid targets
        scores = [(i, self.suspicion[i]) for i in valid]

        # Sort by suspicion (descending - most suspicious first)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Execute most suspicious
        return scores[0][0]

    def get_suspicion_report(self):
        """Get a report of current suspicion levels."""
        return {
            f"P{i}": round(self.suspicion[i], 2)
            for i in range(5)
        }


class MetaAgentStateful(MetaAgent):
    """
    Stateful version that persists suspicion across calls.

    Use this when the agent instance is reused across a single game.
    Call reset_suspicion() at the start of each new game.
    """

    def __init__(self, temperature=1.0, auto_reset=False):
        super().__init__(temperature)
        self.auto_reset = auto_reset
        self._game_id = None

    def get_action(self, obs, action_space=None, **kwargs):
        """Get action, optionally auto-resetting for new games."""
        if self.auto_reset:
            # Detect new game by checking if history is empty
            hist_len = len(obs.get("hist_president", []))
            if hist_len == 0 and self.last_processed_govt >= 0:
                self.reset_suspicion()

        return super().get_action(obs, action_space, **kwargs)


# For backwards compatibility
__all__ = ["MetaAgent", "MetaAgentStateful"]


if __name__ == "__main__":
    # Test the agent
    from game import ShitlerEnv
    from agent import SimpleRandomAgent

    print("Testing MetaAgent...")

    env = ShitlerEnv()
    env.reset(seed=42)

    # Create agents - one meta agent as P0
    meta_agent = MetaAgent()
    random_agent = SimpleRandomAgent()

    agents = {
        "P0": meta_agent,
        "P1": random_agent,
        "P2": random_agent,
        "P3": random_agent,
        "P4": random_agent,
    }

    step = 0
    while not all(env.terminations.values()) and step < 500:
        agent_name = env.agent_selection
        agent_idx = env.agents.index(agent_name)
        obs = env.observe(agent_name)
        action_space = env.action_space(agent_name)
        phase = env.phase

        action = agents[agent_name].get_action(
            obs, action_space, phase=phase, agent_idx=agent_idx
        )
        env.step(action)
        step += 1

        # Print suspicion periodically
        if step % 20 == 0:
            print(f"Step {step}: Suspicion = {meta_agent.get_suspicion_report()}")

    print(f"\nGame ended after {step} steps")
    print(f"Final suspicion: {meta_agent.get_suspicion_report()}")
    print(f"Liberal policies: {env.lib_policies}")
    print(f"Fascist policies: {env.fasc_policies}")

    # Show who won
    for agent, reward in env.rewards.items():
        if reward == 1:
            role = env.roles[agent]
            winner = "Liberals" if role == "lib" else "Fascists"
            print(f"Winner: {winner}")
            break
