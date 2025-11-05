from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

import random

class ShitlerEnv(AECEnv):
    N_LIBS = 3
    N_FASCS = 1
    N_HITTY = 1
    num_agents = N_LIBS + N_FASCS + N_HITTY

    def __init__(self):
        """
        Make a new five-player Secret Hitler game
        """
        super().__init__()
        self.possible_agents = [f"P{n}" for n in range(5)]
        self.possible_roles = ["lib" * 3] + ["fasc"] * 1 + ["hitty"] * 1

    def reset(self, seed=None, options=None):
        """
        Reset the shitler game.
        """
        random.seed(seed)

        self.agents = list(range(self.num_agents))
        self.roles = self.possible_roles[:]
        random.shuffle(self.roles)
        self.rewards = [0 for _ in self.agents]
        self.terminations = [False for _ in self.agents]
        self.truncations = [False for _ in self.agents]
        self.infos = [{} for _ in self.agents]
        self.state = [None for _ in self.agents]
        self.observations = [None for _ in self.agents]
        self.num_moves = 0

        # Possible phases:
        #   nomination, voting, prez_cardsel, chanc_cardsel, prez_decl, chance_decl, special_action
        self.phase = "nomination"

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        Take a turn. Types of actions:
            - Nomination (president)
            - Voting for government (everyone)
            - President card selection (president)
            - Chancellor card selection (chancellor)
            - President declaration (president)
            - Chancellor declaration (chancellor)
            - Special action (president)

        We need to update:
            - rewards
            - terminations
            - truncations
            - infos
            - agent_selection
        and any internal state.
        """
        if (
            self.terminations[self.agent_selection] or
            self.truncations[self.agent_selection]
        ):
            # Agent dead, kill
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection

    def render(self):
        pass

    def observation_space(self, agent):
        return super().observation_space(agent)
    
    def action_space(self, agent):
        return super().action_space(agent)


if __name__ == "__main__":
    env = ShitlerEnv()
    env.reset()
