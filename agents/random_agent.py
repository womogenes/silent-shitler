import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))

from game import ShitlerEnv

class RandomAgent:
    def __init__(self):
        pass

    def get_action(self, obs, action_space):
        """
        Get a random valid action based on observation masks.
        
        For nomination/execution: use the mask from obs
        For card selection: use card_action_mask from obs
        For other phases: sample from action_space directly
        """
        # Check for phase-specific masks
        if "nomination_mask" in obs:
            # Nomination phase: pick from valid nominees
            valid_actions = [i for i, v in enumerate(obs["nomination_mask"]) if v == 1]
            return random.choice(valid_actions)
        
        elif "execution_mask" in obs:
            # Execution phase: pick from valid targets
            valid_actions = [i for i, v in enumerate(obs["execution_mask"]) if v == 1]
            return random.choice(valid_actions)
        
        elif "card_action_mask" in obs:
            # Card selection phase: pick from valid discard options
            valid_actions = [i for i, v in enumerate(obs["card_action_mask"]) if v == 1]
            return random.choice(valid_actions)
        
        else:
            # Voting, claims, etc: sample uniformly
            return action_space.sample()


def play_game(render=True, seed=None):
    env = ShitlerEnv()
    env.reset(seed=seed)
    agents = {agent: RandomAgent() for agent in env.possible_agents}

    if render:
        env.render()

    while not all(env.terminations.values()):
        agent = env.agent_selection
        obs = env.observe(agent)
        action_space = env.action_space(agent)
        action = agents[agent].get_action(obs, action_space)

        env.step(action)

        if render:
            env.render()

    if render:
        env.render()
        print("\nGame over!")
        print(f"Final rewards: {env.rewards}")

    return env.rewards


if __name__ == "__main__":
    play_game(render=True, seed=42)
