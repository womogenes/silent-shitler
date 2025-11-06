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
        self.alive_agents = self.agents[:]

        # Voting
        self.votes = {}

        # Card selection
        self.prez_cards = None
        self.chanc_cards = None
        self.prez_claim = None
        self.chanc_claim = None

        # Phase management
        self.phase = "nomination"

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self.agents[self.president_idx]

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
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
            "chancellor_nominee": self.chancellor_nominee if self.chancellor_nominee is not None else -1,
            "executed": [1 if a in self.executed else 0 for a in self.agents],
            "action_mask": self._get_action_mask(agent),
        }

        # Fascists see all roles
        if self.roles[agent] in ["fasc", "hitty"]:
            obs["all_roles"] = [self._encode_role(self.roles[a]) for a in self.agents]
        else:
            obs["all_roles"] = [-1] * 5

        # Add cards if in card selection phase
        if self.phase == "prez_cardsel" and agent == self.agents[self.president_idx]:
            obs["cards"] = self.prez_cards
        elif self.phase == "chanc_cardsel" and agent == self.agents[self.chancellor_nominee]:
            obs["cards"] = self.chanc_cards
        else:
            obs["cards"] = []

        return obs

    def observation_space(self, agent):
        return spaces.Dict({
            "role": spaces.Discrete(3),  # 0=lib, 1=fasc, 2=hitler
            "lib_policies": spaces.Discrete(6),
            "fasc_policies": spaces.Discrete(7),
            "election_tracker": spaces.Discrete(4),
            "president_idx": spaces.Discrete(5),
            "chancellor_nominee": spaces.Discrete(6),  # -1 to 4
            "executed": spaces.MultiBinary(5),
            "all_roles": spaces.MultiDiscrete([4] * 5),  # -1 to 2
            "cards": spaces.Sequence(spaces.Discrete(2)),
            "action_mask": spaces.Sequence(spaces.Discrete(2)),
        })

    def action_space(self, agent):
        if self.phase == "nomination" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(5)
        elif self.phase == "voting" and agent not in self.executed:
            return spaces.Discrete(2)
        elif self.phase == "prez_cardsel" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(3)
        elif self.phase == "chanc_cardsel" and agent == self.agents[self.chancellor_nominee]:
            return spaces.Discrete(2)
        elif self.phase == "prez_claim" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(4)  # (0,3), (1,2), (2,1), (3,0)
        elif self.phase == "chanc_claim" and agent == self.agents[self.chancellor_nominee]:
            return spaces.Discrete(3)  # (0,2), (1,1), (2,0)
        elif self.phase == "execution" and agent == self.agents[self.president_idx]:
            return spaces.Discrete(5)
        
        return spaces.Discrete(1)

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
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
        self.chancellor_nominee = action
        self.votes = {}
        self.phase = "voting"

    def _handle_voting(self, agent, action):
        self.votes[agent] = action
        if len(self.votes) == len([a for a in self.agents if a not in self.executed]):
            yes_votes = sum(self.votes.values())
            if yes_votes > len(self.votes) // 2:
                self._government_succeeds()
            else:
                self._government_fails()

    def _government_succeeds(self):
        self.election_tracker = 0
        self.last_president = self.president_idx
        self.last_chancellor = self.chancellor_nominee

        # Check Hitler chancellor win (after 3 fasc policies)
        if self.fasc_policies >= 3 and self.roles[self.agents[self.chancellor_nominee]] == "hitty":
            self._end_game("fascists")
            return

        self._draw_cards()
        self.phase = "prez_cardsel"

    def _government_fails(self):
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
        discarded = self.prez_cards.pop(action)
        self.discard.append(discarded)
        self.chanc_cards = self.prez_cards
        self.phase = "chanc_cardsel"

    def _handle_chanc_cardsel(self, action):
        discarded = self.chanc_cards.pop(action)
        self.discard.append(discarded)
        played = self.chanc_cards[0]
        self._play_policy(played)
        self.phase = "prez_claim"

    def _handle_prez_claim(self, action):
        self.prez_claim = action
        self.phase = "chanc_claim"

    def _handle_chanc_claim(self, action):
        self.chanc_claim = action

        # Check if execution needed
        if self.fasc_policies >= 4:
            self.phase = "execution"
        else:
            self._next_president()
            self.phase = "nomination"

    def _handle_execution(self, action):
        target = self.agents[action]
        if target not in self.executed:
            self.executed.add(target)
            self.alive_agents.remove(target)
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
            self._cumulative_rewards[agent] += reward
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

    def _get_action_mask(self, agent):
        """Return binary mask of valid actions for given agent (variable length)"""
        if self.phase == "nomination" and agent == self.agents[self.president_idx]:
            # Can't nominate dead players or players from last government
            mask = []
            for i, a in enumerate(self.agents):
                valid = a not in self.executed
                # Exclude last prez and chanc if they exist
                if self.last_president is not None and i == self.last_president:
                    valid = False
                if self.last_chancellor is not None and i == self.last_chancellor:
                    valid = False
                mask.append(1 if valid else 0)
            return mask

        elif self.phase == "execution" and agent == self.agents[self.president_idx]:
            # Can't execute dead players
            return [0 if a in self.executed else 1 for a in self.agents]

        elif self.phase == "prez_cardsel" and agent == self.agents[self.president_idx]:
            return [1, 1, 1]

        elif self.phase == "chanc_cardsel" and agent == self.agents[self.chancellor_nominee]:
            return [1, 1]

        elif self.phase == "voting":
            return [1, 1]

        elif self.phase == "prez_claim":
            return [1, 1, 1, 1]

        elif self.phase == "chanc_claim":
            return [1, 1, 1]

        else:
            # Default
            return [1]

    def _encode_role(self, role):
        return {"lib": 0, "fasc": 1, "hitty": 2}[role]

    def render(self):
        print("\n" + "="*50)
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
