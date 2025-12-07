from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium import spaces
import numpy as np
import random


class ShitlerEnv(AECEnv):
    metadata = {"name": "shitler_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = [f"P{i}" for i in range(5)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        # Assign roles: 3 libs, 1 fasc, 1 hitler
        roles = ["lib"] * 3 + ["fasc"] + ["hitty"]
        random.shuffle(roles)
        self.roles = {agent: role for agent, role in zip(self.agents, roles)}

        # Game state
        self.lib_policies = 0
        self.fasc_policies = 0
        self.election_tracker = 0
        self.president_idx = 0
        self.chancellor_nominee = None
        self.last_president = None
        self.last_chancellor = None

        # Deck: 0=lib, 1=fasc
        self.deck = [0] * 6 + [1] * 11
        random.shuffle(self.deck)
        self.discard = []

        # Track executed players
        self.executed = set()
        self.votes = {}

        # Card selection
        self.prez_cards = None
        self.chanc_cards = None
        self.prez_claim = None
        self.chanc_claim = None

        # Personal card history (what each player actually saw)
        # Each entry: (government_index, num_libs, num_fascs) or None
        self.personal_cards_seen = {agent: [] for agent in self.agents}

        # Game history (parallel arrays for each government attempt)
        self.hist_president = []
        self.hist_chancellor = []
        self.hist_votes = []  # List of vote arrays [-1=didn't vote, 0=no, 1=yes]
        self.hist_succeeded = []
        self.hist_policy = []  # -1 if government failed
        self.hist_prez_claim = []  # Number of libs claimed (0-3), -1 if failed
        self.hist_chanc_claim = []  # Number of libs claimed (0-2), -1 if failed
        self.hist_execution = []  # Player index executed, -1 if no execution

        # Phase management
        self.phase = "nomination"

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self.agents[self.president_idx]

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.num_moves = 0

    def observe(self, agent):
        if agent not in self.agents:
            return None

        obs = {
            "role": self._encode_role(self.roles[agent]),
            "lib_policies": self.lib_policies,
            "fasc_policies": self.fasc_policies,
            "election_tracker": self.election_tracker,
            "president_idx": self.president_idx,
            "chancellor_nominee": self.chancellor_nominee
            if self.chancellor_nominee is not None
            else -1,
            "executed": [1 if a in self.executed else 0 for a in self.agents],
            # Game history
            "hist_president": self.hist_president[:],
            "hist_chancellor": self.hist_chancellor[:],
            "hist_votes": [v[:] for v in self.hist_votes],
            "hist_succeeded": self.hist_succeeded[:],
            "hist_policy": self.hist_policy[:],
            "hist_prez_claim": self.hist_prez_claim[:],
            "hist_chanc_claim": self.hist_chanc_claim[:],
            "hist_execution": self.hist_execution[:],
        }

        # Fascists see all roles
        if self.roles[agent] in ["fasc", "hitty"]:
            obs["all_roles"] = [self._encode_role(self.roles[a]) for a in self.agents]
        else:
            obs["all_roles"] = [-1] * 5

        # Personal card history (what this player saw in past governments)
        obs["personal_cards_seen"] = self.personal_cards_seen[agent][:]

        # Add phase-specific info
        if self.phase == "nomination" and agent == self.agents[self.president_idx]:
            # Fixed 5-element mask for valid nominees
            valid_indices = self._get_valid_nominees()
            obs["nomination_mask"] = [1 if i in valid_indices else 0 for i in range(5)]

        elif self.phase == "execution" and agent == self.agents[self.president_idx]:
            # Fixed 5-element mask for valid execution targets
            valid_indices = self._get_valid_targets()
            obs["execution_mask"] = [1 if i in valid_indices else 0 for i in range(5)]

        elif self.phase == "prez_cardsel" and agent == self.agents[self.president_idx]:
            # One-hot encoding: (0L,3F), (1L,2F), (2L,1F), (3L,0F)
            num_libs = sum(1 for c in self.prez_cards if c == 0)
            obs["cards"] = [1 if i == num_libs else 0 for i in range(4)]
            # Also provide action mask (can't discard what you don't have)
            obs["card_action_mask"] = [
                1 if num_libs > 0 else 0,
                1 if (3 - num_libs) > 0 else 0
            ]

        elif (
            self.phase == "chanc_cardsel"
            and agent == self.agents[self.chancellor_nominee]
        ):
            # One-hot encoding: (0L,2F), (1L,1F), (2L,0F)
            num_libs = sum(1 for c in self.chanc_cards if c == 0)
            obs["cards"] = [1 if i == num_libs else 0 for i in range(3)]
            # Action mask
            obs["card_action_mask"] = [
                1 if num_libs > 0 else 0,
                1 if (2 - num_libs) > 0 else 0
            ]

        return obs

    def action_space(self, agent):
        # Fixed action spaces for consistent semantics
        if self.phase == "nomination" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(5)  # Always 5 (one per player), masked externally

        elif self.phase == "voting" and agent not in self.executed:
            return spaces.Discrete(2)  # 0=no, 1=yes

        elif self.phase == "prez_cardsel" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(2)  # 0=discard liberal, 1=discard fascist

        elif (
            self.phase == "chanc_cardsel"
            and agent == self.agents[self.chancellor_nominee]
        ):
            return spaces.Discrete(2)  # 0=discard liberal, 1=discard fascist
        
        elif self.phase == "prez_claim" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(4)  # (0L,3F), (1L,2F), (2L,1F), (3L,0F)
        
        elif (
            self.phase == "chanc_claim"
            and agent == self.agents[self.chancellor_nominee]
        ):
            return spaces.Discrete(3)  # (0L,2F), (1L,1F), (2L,0F)
        
        elif self.phase == "execution" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(5)  # Always 5 (one per player), masked externally

        return spaces.Discrete(1)

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        if self.phase == "nomination":
            self._handle_nomination(action)

        elif self.phase == "voting":
            self._handle_voting(agent, action)

        elif self.phase == "prez_cardsel":
            self._handle_prez_cardsel(action)

        elif self.phase == "chanc_cardsel":
            self._handle_chanc_cardsel(action)

        elif self.phase == "prez_claim":
            self._handle_prez_claim(action)

        elif self.phase == "chanc_claim":
            self._handle_chanc_claim(action)

        elif self.phase == "execution":
            self._handle_execution(action)

        self.num_moves += 1
        self._check_game_end()

        if not any(self.terminations.values()):
            self._update_agent_selection()

    def _handle_nomination(self, action):
        # Action is now a player index directly (0-4)
        # Caller must ensure action is valid (in nomination_mask)
        valid_nominees = self._get_valid_nominees()
        if action not in valid_nominees:
            raise ValueError(f"Invalid nomination: player {action} not in valid nominees {valid_nominees}")
        self.chancellor_nominee = action
        self.votes = {}
        self.phase = "voting"

    def _handle_voting(self, agent, action):
        self.votes[agent] = action
        if len(self.votes) == sum(1 for a in self.agents if a not in self.executed):
            # All votes are in - record this government attempt
            self._record_government_attempt()
            yes_votes = sum(self.votes.values())
            if yes_votes > len(self.votes) // 2:
                self._government_succeeds()
            else:
                self._government_fails()

    def _government_succeeds(self):
        self.hist_succeeded[-1] = 1
        self.election_tracker = 0
        self.last_president = self.president_idx
        self.last_chancellor = self.chancellor_nominee

        # Check Hitler chancellor win (after 3 fasc policies)
        if (
            self.fasc_policies >= 3
            and self.roles[self.agents[self.chancellor_nominee]] == "hitty"
        ):
            self._end_game("fascists")
            return

        self._draw_cards()
        self.phase = "prez_cardsel"

    def _government_fails(self):
        self.hist_succeeded[-1] = 0
        # Record -1 for failed government fields
        self.hist_policy[-1] = -1
        self.hist_prez_claim[-1] = -1
        self.hist_chanc_claim[-1] = -1
        self.hist_execution[-1] = -1

        self.election_tracker += 1
        if self.election_tracker >= 3:
            self._chaos()
        else:
            self._next_president()
            self.phase = "nomination"

    def _chaos(self):
        self.election_tracker = 0
        if len(self.deck) < 1:
            self._reshuffle()
        card = self.deck.pop(0)
        self._play_policy(card)
        self._next_president()
        self.phase = "nomination"

    def _draw_cards(self):
        if len(self.deck) < 3:
            self._reshuffle()
        self.prez_cards = [self.deck.pop(0) for _ in range(3)]

    def _handle_prez_cardsel(self, action):
        # Action: 0=discard liberal, 1=discard fascist
        num_libs = sum(1 for c in self.prez_cards if c == 0)
        num_fascs = 3 - num_libs
        
        # Record what president saw before discarding
        prez_agent = self.agents[self.president_idx]
        gov_idx = len(self.hist_president) - 1
        self.personal_cards_seen[prez_agent].append((gov_idx, num_libs, num_fascs))
        
        # Validate action
        if action == 0 and num_libs == 0:
            raise ValueError("Cannot discard liberal: no liberals in hand")
        if action == 1 and num_fascs == 0:
            raise ValueError("Cannot discard fascist: no fascists in hand")
        
        # Find and remove the card
        discard_type = 0 if action == 0 else 1  # 0=lib, 1=fasc
        for i, card in enumerate(self.prez_cards):
            if card == discard_type:
                discarded = self.prez_cards.pop(i)
                self.discard.append(discarded)
                break
        
        self.chanc_cards = self.prez_cards
        self.phase = "chanc_cardsel"

    def _handle_chanc_cardsel(self, action):
        # Action: 0=discard liberal (play fascist), 1=discard fascist (play liberal)
        num_libs = sum(1 for c in self.chanc_cards if c == 0)
        num_fascs = 2 - num_libs
        
        # Record what chancellor saw before discarding
        chanc_agent = self.agents[self.chancellor_nominee]
        gov_idx = len(self.hist_president) - 1
        self.personal_cards_seen[chanc_agent].append((gov_idx, num_libs, num_fascs))
        
        # Validate action
        if action == 0 and num_libs == 0:
            raise ValueError("Cannot discard liberal: no liberals in hand")
        if action == 1 and num_fascs == 0:
            raise ValueError("Cannot discard fascist: no fascists in hand")
        
        # Find and remove the card
        discard_type = 0 if action == 0 else 1
        for i, card in enumerate(self.chanc_cards):
            if card == discard_type:
                discarded = self.chanc_cards.pop(i)
                self.discard.append(discarded)
                break
        
        played = self.chanc_cards[0]
        if played == 0:
            self.lib_policies += 1
        else:
            self.fasc_policies += 1
        self.hist_policy[-1] = played
        self.phase = "prez_claim"

    def _handle_prez_claim(self, action):
        self.prez_claim = action
        self.hist_prez_claim[-1] = action  # Action is number of libs (0-3)
        self.phase = "chanc_claim"

    def _handle_chanc_claim(self, action):
        self.chanc_claim = action
        self.hist_chanc_claim[-1] = action  # Action is number of libs (0-2)

        # Check if execution needed
        if self.fasc_policies >= 4:
            self.phase = "execution"
        else:
            self.hist_execution[-1] = -1  # No execution
            self._next_president()
            self.phase = "nomination"

    def _handle_execution(self, action):
        # Action is now a player index directly (0-4)
        valid_targets = self._get_valid_targets()
        if action not in valid_targets:
            raise ValueError(f"Invalid execution: player {action} not in valid targets {valid_targets}")
        target = self.agents[action]
        self.executed.add(target)
        self.hist_execution[-1] = action
        if self.roles[target] == "hitty":
            self._end_game("liberals")
            return
        self._next_president()
        self.phase = "nomination"

    def _play_policy(self, card):
        if card == 0:
            self.lib_policies += 1
        else:
            self.fasc_policies += 1

    def _next_president(self):
        self.president_idx = (self.president_idx + 1) % 5
        while self.agents[self.president_idx] in self.executed:
            self.president_idx = (self.president_idx + 1) % 5

    def _reshuffle(self):
        self.deck.extend(self.discard)
        self.discard = []
        random.shuffle(self.deck)

    def _check_game_end(self):
        # Check if too few players remain
        alive_count = sum(1 for a in self.agents if a not in self.executed)
        if alive_count < 2:
            # Game cannot continue - fascists win by attrition
            self._end_game("fascists")
            return

        if self.lib_policies >= 5:
            self._end_game("liberals")
        elif self.fasc_policies >= 6:
            self._end_game("fascists")

    def _end_game(self, winner):
        for agent in self.agents:
            if winner == "liberals":
                reward = 1 if self.roles[agent] == "lib" else -1
            else:
                reward = 1 if self.roles[agent] in ["fasc", "hitty"] else -1
            self.rewards[agent] = reward
            self.terminations[agent] = True

    def _update_agent_selection(self):
        if self.phase == "voting":
            # Round-robin through alive agents
            current_idx = self.agents.index(self.agent_selection)
            next_idx = (current_idx + 1) % 5
            while self.agents[next_idx] in self.executed:
                next_idx = (next_idx + 1) % 5
            self.agent_selection = self.agents[next_idx]

        elif self.phase == "nomination":
            self.agent_selection = self.agents[self.president_idx]

        elif self.phase == "prez_cardsel" or self.phase == "prez_claim":
            self.agent_selection = self.agents[self.president_idx]

        elif self.phase == "chanc_cardsel" or self.phase == "chanc_claim":
            self.agent_selection = self.agents[self.chancellor_nominee]

        elif self.phase == "execution":
            self.agent_selection = self.agents[self.president_idx]

    def _record_government_attempt(self):
        """Record a government attempt after all votes are in"""
        # Convert votes dict to array with -1 for dead/missing players
        vote_array = []
        for agent in self.agents:
            if agent in self.executed:
                vote_array.append(-1)
            else:
                vote_array.append(self.votes.get(agent, 0))

        self.hist_president.append(self.president_idx)
        self.hist_chancellor.append(self.chancellor_nominee)
        self.hist_votes.append(vote_array)
        self.hist_succeeded.append(-1)  # Will be set to 0 or 1
        self.hist_policy.append(-1)  # Will be set if succeeded
        self.hist_prez_claim.append(-1)  # Will be set if succeeded
        self.hist_chanc_claim.append(-1)  # Will be set if succeeded
        self.hist_execution.append(-1)  # Will be set if execution happens

    def _get_valid_nominees(self):
        """Return list of player indices that can be nominated as chancellor"""
        alive_count = sum(1 for a in self.agents if a not in self.executed)

        # Edge case: With only 2 players, must be able to nominate the other one
        if alive_count == 2:
            return [
                i
                for i, a in enumerate(self.agents)
                if a not in self.executed and i != self.president_idx
            ]

        # Only exclude last president/chancellor if they're still alive
        exclude_last_prez = (
            self.last_president is not None
            and self.agents[self.last_president] not in self.executed
        )
        exclude_last_chanc = (
            self.last_chancellor is not None
            and self.agents[self.last_chancellor] not in self.executed
        )

        # Term limits are relaxed when fewer than 5 players are alive
        # Only the immediate previous chancellor is ineligible (if alive)
        if alive_count < 5:
            nominees = [
                i
                for i, a in enumerate(self.agents)
                if a not in self.executed
                and i != self.president_idx  # Can't nominate yourself
                and (not exclude_last_chanc or i != self.last_chancellor)
            ]
        else:
            # Normal term limits: can't nominate last president or chancellor (if alive)
            nominees = [
                i
                for i, a in enumerate(self.agents)
                if a not in self.executed
                and i != self.president_idx  # Can't nominate yourself
                and (not exclude_last_prez or i != self.last_president)
                and (not exclude_last_chanc or i != self.last_chancellor)
            ]

        return nominees

    def _get_valid_targets(self):
        """Return list of player indices that can be executed"""
        return [i for i, a in enumerate(self.agents) if a not in self.executed]

    def _encode_role(self, role):
        return {"lib": 0, "fasc": 1, "hitty": 2}[role]

    def render(self):
        print("\n" + "=" * 50)
        print(f"SILENT SHITLER - Move {self.num_moves}")
        print("=" * 50)
        print(f"Liberal policies: {self.lib_policies}/5")
        print(f"Fascist policies: {self.fasc_policies}/6")
        print(f"Election tracker: {self.election_tracker}/3")
        print(f"Deck: {len(self.deck)}/17 cards, Discard: {len(self.discard)}/17 cards")
        print()
        print(f"President: P{self.president_idx}")
        if self.chancellor_nominee is not None:
            print(f"Chancellor nominee: P{self.chancellor_nominee}")
        print(f"Phase: {self.phase}")
        print()
        print("Players:")
        for i, agent in enumerate(self.agents):
            status = "EXECUTED" if agent in self.executed else "alive"
            role = self.roles[agent].upper()
            print(f"  {agent} ({role:>6}) {status}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    env = ShitlerEnv()
    env.reset(seed=42)
    env.render()
