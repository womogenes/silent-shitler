import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, Callable

sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))
from game import ShitlerEnv


def run_games(agent_factory: Callable, num_games: int, seed: int = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Run multiple games and collect statistics.

    Args:
        agent_factory: Function that returns a new agent instance
        num_games: Number of games to run
        seed: Random seed for reproducibility
        verbose: If True, print progress

    Returns:
        Dictionary with evaluation metrics
    """
    liberal_rewards = []
    fascist_rewards = []
    liberal_wins = 0
    fascist_wins = 0
    game_lengths = []

    # Track win conditions
    win_conditions = {
        "lib_5_policies": 0,
        "fasc_6_policies": 0,
        "hitler_chancellor": 0,
        "hitler_executed": 0,
    }

    for game_idx in range(num_games):
        if verbose and (game_idx + 1) % 100 == 0:
            print(f"Progress: {game_idx + 1}/{num_games} games")

        game_seed = seed + game_idx if seed is not None else None
        rewards, moves, win_condition = play_single_game(agent_factory, game_seed)

        # Track game length
        game_lengths.append(moves)

        # Track win conditions
        if win_condition:
            win_conditions[win_condition] += 1

        # Separate rewards by team
        lib_reward = rewards["lib"]
        fasc_reward = rewards["fasc"]

        liberal_rewards.append(lib_reward)
        fascist_rewards.append(fasc_reward)

        if lib_reward > 0:
            liberal_wins += 1
        else:
            fascist_wins += 1

    # Compute statistics (convert numpy types to Python native types for JSON serialization)
    results = {
        "num_games": num_games,
        "liberal_win_rate": float(liberal_wins / num_games),
        "fascist_win_rate": float(fascist_wins / num_games),
        "liberal_avg_reward": float(np.mean(liberal_rewards)),
        "fascist_avg_reward": float(np.mean(fascist_rewards)),
        "liberal_reward_std": float(np.std(liberal_rewards)),
        "fascist_reward_std": float(np.std(fascist_rewards)),
        "avg_game_length": float(np.mean(game_lengths)),
        "game_length_std": float(np.std(game_lengths)),
        "min_game_length": int(np.min(game_lengths)),
        "max_game_length": int(np.max(game_lengths)),
        "win_conditions": win_conditions,
    }

    return results


def play_single_game(agent_factory: Callable, seed: int = None) -> tuple:
    """
    Play a single game to completion.

    Args:
        agent_factory: Function that takes env and returns a new agent instance
        seed: Random seed

    Returns:
        Tuple of (rewards_by_team, num_moves, win_condition)
    """
    env = ShitlerEnv()
    env.reset(seed=seed)
    # Agent factory can take env as optional arg for agents that need it
    try:
        agents = {agent: agent_factory(env) for agent in env.possible_agents}
    except TypeError:
        # Fallback for simple agents that don't need env
        agents = {agent: agent_factory() for agent in env.possible_agents}

    # Track initial roles for reward aggregation
    roles = env.roles.copy()

    while not all(env.terminations.values()):
        agent = env.agent_selection
        obs = env.observe(agent)
        action_space = env.action_space(agent)
        action = agents[agent].get_action(obs, action_space)
        env.step(action)

    # Aggregate rewards by team (average across team members)
    lib_agents = [a for a, r in roles.items() if r == "lib"]
    fasc_agents = [a for a, r in roles.items() if r in ["fasc", "hitty"]]

    lib_reward = np.mean([env.rewards[a] for a in lib_agents])
    fasc_reward = np.mean([env.rewards[a] for a in fasc_agents])

    # Determine win condition
    win_condition = determine_win_condition(env)

    return {"lib": lib_reward, "fasc": fasc_reward}, env.num_moves, win_condition


def determine_win_condition(env: ShitlerEnv) -> str:
    """Determine how the game was won."""
    if env.lib_policies >= 5:
        return "lib_5_policies"
    elif env.fasc_policies >= 6:
        return "fasc_6_policies"
    else:
        # Check if Hitler was executed or elected
        # We need to infer from game state
        hitler_agent = [a for a, r in env.roles.items() if r == "hitty"][0]
        if hitler_agent in env.executed:
            return "hitler_executed"
        else:
            # If fascists won but didn't reach 6 policies, Hitler must have been elected
            if any(env.rewards[a] > 0 for a, r in env.roles.items() if r in ["fasc", "hitty"]):
                return "hitler_chancellor"
    return None


def print_results(results: Dict[str, Any]):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Games played: {results['num_games']}")
    print()
    print("WIN RATES:")
    print(f"  Liberals:  {results['liberal_win_rate']:.2%}")
    print(f"  Fascists:  {results['fascist_win_rate']:.2%}")
    print()
    print("AVERAGE REWARDS:")
    print(f"  Liberals:  {results['liberal_avg_reward']:+.3f} ± {results['liberal_reward_std']:.3f}")
    print(f"  Fascists:  {results['fascist_avg_reward']:+.3f} ± {results['fascist_reward_std']:.3f}")
    print()
    print("GAME LENGTH:")
    print(f"  Average:   {results['avg_game_length']:.1f} ± {results['game_length_std']:.1f} moves")
    print(f"  Range:     {results['min_game_length']} - {results['max_game_length']} moves")
    print()
    print("WIN CONDITIONS:")
    for condition, count in results['win_conditions'].items():
        percentage = count / results['num_games'] * 100
        condition_name = condition.replace("_", " ").title()
        print(f"  {condition_name:25s}: {count:5d} ({percentage:5.2f}%)")
    print("=" * 60 + "\n")
