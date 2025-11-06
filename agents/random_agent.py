import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))

from game import ShitlerEnv

class RandomAgent:
    def __init__(self):
        pass

    def get_action(self, obs, action_space):
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
