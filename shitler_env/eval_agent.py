"""
Run evaluation games against several Agent classes
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from game import ShitlerEnv


def evaluate_agents(agent_classes, num_games=100, verbose=False, seed=None):
    """
    Run evaluation games with specified agents.

    Args:
        agent_classes: List of 5 agent classes (one per player)
        num_games: Number of games to simulate
        verbose: Whether to print progress
        seed: Random seed for reproducibility

    Returns:
        dict: Results with win rates and statistics
    """
    if len(agent_classes) != 5:
        raise ValueError(f"Expected 5 agent classes, got {len(agent_classes)}")

    # Track results
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "player_rewards": {f"P{i}": [] for i in range(5)},
    }

    for game_num in range(num_games):
        # Create environment and agents
        env = ShitlerEnv()
        game_seed = None if seed is None else seed + game_num
        env.reset(seed=game_seed)

        # Instantiate agents
        agents = {
            f"P{i}": agent_classes[i]()
            for i in range(5)
        }

        # Play game
        while not all(env.terminations.values()):
            agent_name = env.agent_selection
            obs = env.observe(agent_name)
            action_space = env.action_space(agent_name)

            # Get action from agent
            action = agents[agent_name].get_action(obs, action_space)
            env.step(action)

        # Record all player rewards
        for agent_name, reward in env.rewards.items():
            results["player_rewards"][agent_name].append(reward)

        # Determine winning team (check any player with reward == 1)
        for agent_name, reward in env.rewards.items():
            if reward == 1:
                role = env.roles[agent_name]
                if role == "lib":
                    results["lib_wins"] += 1
                else:
                    results["fasc_wins"] += 1
                break  # Only count once per game

        if verbose and (game_num + 1) % 10 == 0:
            print(f"Completed {game_num + 1}/{num_games} games")

    # Calculate statistics
    results["lib_win_rate"] = results["lib_wins"] / num_games
    results["fasc_win_rate"] = results["fasc_wins"] / num_games
    results["num_games"] = num_games

    # Average rewards per player
    results["avg_rewards"] = {
        player: sum(rewards) / len(rewards)
        for player, rewards in results["player_rewards"].items()
    }

    return results


if __name__ == "__main__":
    # Example usage with SimpleRandomAgent
    from agent import SimpleRandomAgent

    # Run evaluation with 5 random agents
    agents = [SimpleRandomAgent for _ in range(5)]
    results = evaluate_agents(agents, num_games=1000, verbose=True, seed=42)

    print("\nResults:")
    print(f"Liberal win rate: {results['lib_win_rate']:.2%}")
    print(f"Fascist win rate: {results['fasc_win_rate']:.2%}")
    print(f"Average rewards: {results['avg_rewards']}")
