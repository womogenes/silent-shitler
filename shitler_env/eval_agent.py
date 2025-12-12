"""
Run evaluation games with parallel execution and factory-based agent construction.
"""

import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from game import ShitlerEnv

# Public API
__all__ = ['evaluate_agents', 'AgentFactory']


class AgentFactory:
    """Helper class to create agents in worker processes (pickleable)."""
    def __init__(self, agent_class, agent_kwargs):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs

    def create(self):
        """Create a new agent instance."""
        return self.agent_class(**self.agent_kwargs)


def _run_game_chunk_wrapper(args):
    """Wrapper to unpack arguments for imap_unordered."""
    return _run_game_chunk(*args)


def _run_game_chunk(game_nums, agent_factories, base_seed, track_win_reasons):
    """Run multiple games in a chunk, reusing agent instances.

    Args:
        game_nums: List of game numbers to run
        agent_factories: List of AgentFactory instances (5 agents)
        base_seed: Base random seed
        track_win_reasons: Whether to track detailed win reasons

    Returns:
        List of game results
    """
    # Create agents ONCE for this worker (expensive operation like loading networks)
    agents = [factory.create() for factory in agent_factories]

    # Run multiple games, reusing the same agent instances
    chunk_results = []

    for game_num in game_nums:
        # Create environment
        env = ShitlerEnv()
        game_seed = None if base_seed is None else base_seed + game_num
        env.reset(seed=game_seed)

        # Reset agents for new game (cheap operation)
        game_agents = {}
        for i, agent in enumerate(agents):
            if hasattr(agent, 'reset'):
                agent.reset(player_idx=i)
            game_agents[f"P{i}"] = agent

        # Play game
        while not all(env.terminations.values()):
            agent_name = env.agent_selection
            obs = env.observe(agent_name)
            action_space = env.action_space(agent_name)
            agent = game_agents[agent_name]

            # Check if this is a DeepRoleAgentV2 that needs game_state
            if hasattr(agent, '__class__') and agent.__class__.__name__ == 'DeepRoleAgentV2':
                game_state = env.get_state_dict()
                action = agent.get_action(obs, game_state=game_state, agent_name=agent_name)
            else:
                action = agent.get_action(obs, action_space)

            env.step(action)

        # Collect results from this game
        game_result = {
            "rewards": dict(env.rewards),
            "roles": dict(env.roles),
        }

        # Add game state for win reason tracking
        if track_win_reasons:
            game_result["lib_policies"] = env.lib_policies
            game_result["fasc_policies"] = env.fasc_policies
            game_result["phase"] = env.phase
            # Check if Hitler was elected chancellor
            if hasattr(env, 'last_chancellor') and env.last_chancellor is not None:
                chancellor_agent = env.agents[env.last_chancellor]
                game_result["last_chancellor_was_hitler"] = (env.roles[chancellor_agent] == "hitty")
            else:
                game_result["last_chancellor_was_hitler"] = False

        chunk_results.append(game_result)

    return chunk_results


def evaluate_agents(agent_factories, num_games=100, seed=None, track_win_reasons=False, num_workers=-1, verbose=True):
    """
    Run evaluation games in parallel with factory-based agent construction.

    Args:
        agent_factories: List of 5 AgentFactory instances (one per player position)
        num_games: Number of games to simulate
        seed: Random seed for reproducibility
        track_win_reasons: Whether to track win reasons (e.g., "5 policies", "Hitler executed")
        num_workers: Number of parallel workers. -1 = all CPU cores, or specify a number
        verbose: Whether to print progress

    Returns:
        dict: Results with win rates and statistics
    """
    # Validate inputs
    if len(agent_factories) != 5:
        raise ValueError(f"agent_factories must contain exactly 5 factories, got {len(agent_factories)}")

    # Determine actual number of workers
    if num_workers == -1:
        num_workers = cpu_count()

    if verbose:
        print(f"Running {num_games} games in parallel with {num_workers} workers...")

    # Track results
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "liberal_wins": 0,
        "fascist_wins": 0,
    }

    if track_win_reasons:
        results["lib_win_reasons"] = {}
        results["fasc_win_reasons"] = {}

    # Divide games into chunks for workers to process
    # Each worker will create agents once and reuse them across multiple games
    games_per_chunk = max(1, num_games // num_workers)
    game_chunks = []

    for worker_idx in range(num_workers):
        start_game = worker_idx * games_per_chunk
        if worker_idx == num_workers - 1:
            # Last worker gets any remaining games
            end_game = num_games
        else:
            end_game = start_game + games_per_chunk

        if start_game < num_games:
            game_nums = list(range(start_game, end_game))
            game_chunks.append((game_nums, agent_factories, seed, track_win_reasons))

    if verbose:
        print(f"Split into {len(game_chunks)} chunks (~{games_per_chunk} games per worker)")

    # Run game chunks in parallel
    with Pool(num_workers) as pool:
        if verbose:
            # Use imap_unordered for progress tracking with live stats
            pbar = tqdm(total=num_games, ncols=100, desc="Lib: 0 | Fasc: 0")
            game_results = []
            lib_wins_so_far = 0
            fasc_wins_so_far = 0

            for chunk_results in pool.imap_unordered(_run_game_chunk_wrapper, game_chunks):
                # Each chunk returns a list of game results
                for game_result in chunk_results:
                    game_results.append(game_result)

                    # Update win counts
                    for agent_name, reward in game_result["rewards"].items():
                        if reward == 1:
                            role = game_result["roles"][agent_name]
                            if role == "lib":
                                lib_wins_so_far += 1
                            else:
                                fasc_wins_so_far += 1
                            break

                    # Update progress bar with current win counts
                    pbar.set_description(f"Lib: {lib_wins_so_far} | Fasc: {fasc_wins_so_far}")
                    pbar.update(1)

            pbar.close()
        else:
            all_chunk_results = pool.starmap(_run_game_chunk, game_chunks)
            # Flatten the list of lists
            game_results = [game_result for chunk_results in all_chunk_results for game_result in chunk_results]

    # Aggregate results
    for game_result in game_results:
        # Determine winning team
        for agent_name, reward in game_result["rewards"].items():
            if reward == 1:
                role = game_result["roles"][agent_name]
                if role == "lib":
                    results["lib_wins"] += 1
                    results["liberal_wins"] += 1

                    # Track win reason if requested
                    if track_win_reasons:
                        if game_result["lib_policies"] >= 5:
                            reason = "5 policies"
                        else:
                            reason = "Hitler executed"
                        results["lib_win_reasons"][reason] = results["lib_win_reasons"].get(reason, 0) + 1
                else:
                    results["fasc_wins"] += 1
                    results["fascist_wins"] += 1

                    # Track win reason if requested
                    if track_win_reasons:
                        if game_result["fasc_policies"] >= 6:
                            reason = "6 policies"
                        elif game_result.get("last_chancellor_was_hitler", False):
                            reason = "Hitler chancellor"
                        else:
                            # This shouldn't happen if game logic is correct
                            if verbose:
                                print(f"WARNING: Fascist win with unclear reason - policies: {game_result['fasc_policies']}, "
                                      f"hitler chancellor: {game_result.get('last_chancellor_was_hitler', False)}, "
                                      f"phase: {game_result.get('phase', 'unknown')}")
                            reason = "other"

                        results["fasc_win_reasons"][reason] = results["fasc_win_reasons"].get(reason, 0) + 1
                break

    # Calculate statistics
    results["lib_win_rate"] = results["lib_wins"] / num_games
    results["fasc_win_rate"] = results["fasc_wins"] / num_games
    results["num_games"] = num_games

    return results
