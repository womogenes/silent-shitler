"""
Run evaluation games against several Agent classes
"""

import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from game import ShitlerEnv

# Public API
__all__ = ['evaluate_agents', '_AgentFactory']


class _AgentFactory:
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
    """Run multiple games in a chunk, reusing agent instances. Helper for parallel execution.

    Args:
        game_nums: List of game numbers to run
        agent_factories: List of _AgentFactory instances to create agents
        base_seed: Base random seed
        track_win_reasons: Whether to track detailed win reasons

    Returns:
        List of game results
    """
    # Create agents ONCE for this worker (expensive operation)
    agents = []
    for i in range(5):
        agent = agent_factories[i].create()
        agents.append(agent)

    # Run multiple games, reusing the same agent instances
    chunk_results = []

    for game_num in game_nums:
        # Create environment
        env = ShitlerEnv()
        game_seed = None if base_seed is None else base_seed + game_num
        env.reset(seed=game_seed)

        # Reset agents for new game (much cheaper than creating new instances)
        game_agents = {}
        for i in range(5):
            agent = agents[i]
            if hasattr(agent, 'reset'):
                agent.reset(player_idx=i)
            game_agents[f"P{i}"] = agent

        # Play game
        while not all(env.terminations.values()):
            agent_name = env.agent_selection
            obs = env.observe(agent_name)
            action_space = env.action_space(agent_name)

            # Get action from agent - check if it needs special handling
            agent = game_agents[agent_name]

            # Check if this is a DeepRoleAgentV2 that needs game_state
            if hasattr(agent, '__class__') and agent.__class__.__name__ == 'DeepRoleAgentV2':
                # Pass game state for DeepRoleAgentV2
                game_state = env.get_state_dict()
                action = agent.get_action(obs, game_state=game_state, agent_name=agent_name)
            else:
                # Standard agent interface
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
            # Check if Hitler was elected chancellor (for win reason tracking)
            if hasattr(env, 'last_chancellor') and env.last_chancellor is not None:
                chancellor_agent = env.agents[env.last_chancellor]
                game_result["last_chancellor_was_hitler"] = (env.roles[chancellor_agent] == "hitty")
            else:
                game_result["last_chancellor_was_hitler"] = False

        chunk_results.append(game_result)

    return chunk_results


def evaluate_agents(agents, num_games=100, verbose=False, seed=None, lib_agents=None, fasc_agents=None, track_win_reasons=False, num_workers=None, agent_factories=None):
    """
    Run evaluation games with specified agents.

    Args:
        agents: Either:
            - List of 5 agent instances (one per player)
            - Dict mapping player indices (0-4) to agent instances
            - None if using lib_agents and fasc_agents
        num_games: Number of games to simulate
        verbose: Whether to print progress
        seed: Random seed for reproducibility
        lib_agents: Optional list of liberal agent instances (will be assigned randomly)
        fasc_agents: Optional list of fascist/hitler agent instances (will be assigned randomly)
        track_win_reasons: Whether to track win reasons (e.g., "5 policies", "Hitler executed")
        num_workers: Number of parallel workers to use. If None, runs sequentially.
                     Set to -1 to use all available CPU cores.
        agent_factories: Optional list of 5 callables that create agent instances for parallel execution.
                        Each callable should return a fresh agent instance. Required for parallel execution.
                        Example: [lambda: DeepRoleAgent(...), lambda: MetaAgent(...), ...]

    Returns:
        dict: Results with win rates and statistics
    """
    # Handle different input formats
    if agents is not None:
        # Convert dict to list if needed
        if isinstance(agents, dict):
            if len(agents) != 5 or set(agents.keys()) != set(range(5)):
                raise ValueError(f"Expected dict with keys 0-4, got {agents.keys()}")
            agents_list = [agents[i] for i in range(5)]
        else:
            if len(agents) != 5:
                raise ValueError(f"Expected 5 agents, got {len(agents)}")
            agents_list = agents

        # Verify all are instances, not classes
        for i, agent in enumerate(agents_list):
            if isinstance(agent, type):
                raise TypeError(f"Agent at index {i} is a class, not an instance. Please instantiate all agents before passing them.")
    elif lib_agents is not None and fasc_agents is not None:
        # Verify all are instances
        for i, agent in enumerate(lib_agents):
            if isinstance(agent, type):
                raise TypeError(f"Liberal agent at index {i} is a class, not an instance. Please instantiate all agents before passing them.")
        for i, agent in enumerate(fasc_agents):
            if isinstance(agent, type):
                raise TypeError(f"Fascist agent at index {i} is a class, not an instance. Please instantiate all agents before passing them.")
        # Will handle random assignment below
        agents_list = None
    else:
        raise ValueError("Must provide either agents or both lib_agents and fasc_agents")

    # Track results
    results = {
        "lib_wins": 0,
        "fasc_wins": 0,
        "player_rewards": {f"P{i}": [] for i in range(5)},
        "liberal_wins": 0,
        "fascist_wins": 0,
        "win_reasons": {},
    }

    # Add detailed win reason tracking if requested
    if track_win_reasons:
        results["lib_win_reasons"] = {}
        results["fasc_win_reasons"] = {}

    # Handle parallel execution
    if num_workers is not None and num_workers != 0:
        if agents_list is None:
            raise ValueError("Parallel execution (num_workers) currently only supports fixed agent lists, not lib_agents/fasc_agents")

        if agent_factories is None:
            raise ValueError(
                "Parallel execution requires agent_factories parameter. "
                "Please provide a list of _AgentFactory instances or use num_workers=None for sequential execution. "
                "Example: agent_factories=[_AgentFactory(DeepRoleAgentV2, {'networks_path': '...'}), ...]"
            )

        if len(agent_factories) != 5:
            raise ValueError(f"agent_factories must contain exactly 5 factories, got {len(agent_factories)}")

        # Determine actual number of workers
        if num_workers == -1:
            num_workers = cpu_count()

        if verbose:
            print(f"Running {num_games} games in parallel with {num_workers} workers...")

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
            # Record player rewards
            for agent_name, reward in game_result["rewards"].items():
                results["player_rewards"][agent_name].append(reward)

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
                                print(f"WARNING: Fascist win with unclear reason - policies: {game_result['fasc_policies']}, "
                                      f"hitler chancellor: {game_result.get('last_chancellor_was_hitler', False)}, "
                                      f"phase: {game_result.get('phase', 'unknown')}")
                                reason = "other"

                            results["fasc_win_reasons"][reason] = results["fasc_win_reasons"].get(reason, 0) + 1
                    break  # Only count once per game

    else:
        # Sequential execution (original code)
        for game_num in tqdm(range(num_games), ncols=80, disable=not verbose):
            # Create environment
            env = ShitlerEnv()
            game_seed = None if seed is None else seed + game_num
            env.reset(seed=game_seed)

            # Reset agents for new game
            if agents_list is not None:
                # Use fixed agent list
                game_agents = {}
                for i in range(5):
                    agent = agents_list[i]
                    # Reset the agent if it has a reset method
                    if hasattr(agent, 'reset'):
                        agent.reset(player_idx=i)
                    game_agents[f"P{i}"] = agent
            else:
                # Random assignment of lib/fasc agents
                import random
                if game_seed is not None:
                    random.seed(game_seed)

                # Randomly assign roles
                indices = list(range(5))
                random.shuffle(indices)
                lib_indices = indices[:3]

                game_agents = {}
                lib_idx = 0
                fasc_idx = 0

                for i in range(5):
                    if i in lib_indices:
                        agent = lib_agents[lib_idx % len(lib_agents)]
                        lib_idx += 1
                    else:
                        agent = fasc_agents[fasc_idx % len(fasc_agents)]
                        fasc_idx += 1

                    # Reset the agent if it has a reset method
                    if hasattr(agent, 'reset'):
                        agent.reset(player_idx=i)
                    game_agents[f"P{i}"] = agent

            # Play game
            while not all(env.terminations.values()):
                agent_name = env.agent_selection
                obs = env.observe(agent_name)
                action_space = env.action_space(agent_name)

                # Get action from agent - check if it needs special handling
                agent = game_agents[agent_name]

                # Check if this is a DeepRoleAgentV2 that needs game_state
                if hasattr(agent, '__class__') and agent.__class__.__name__ == 'DeepRoleAgentV2':
                    # Pass game state for DeepRoleAgentV2
                    game_state = env.get_state_dict()
                    action = agent.get_action(obs, game_state=game_state, agent_name=agent_name)
                else:
                    # Standard agent interface
                    action = agent.get_action(obs, action_space)

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
                        results["liberal_wins"] += 1

                        # Track win reason if requested
                        if track_win_reasons:
                            if env.lib_policies >= 5:
                                reason = "5 policies"
                            else:
                                reason = "Hitler executed"
                            results["lib_win_reasons"][reason] = results["lib_win_reasons"].get(reason, 0) + 1
                    else:
                        results["fasc_wins"] += 1
                        results["fascist_wins"] += 1

                        # Track win reason if requested
                        if track_win_reasons:
                            if env.fasc_policies >= 6:
                                reason = "6 policies"
                            elif hasattr(env, 'last_chancellor') and env.last_chancellor is not None:
                                chancellor_agent = env.agents[env.last_chancellor]
                                if env.roles[chancellor_agent] == "hitty":
                                    reason = "Hitler chancellor"
                                else:
                                    print(f"WARNING: Fascist win with unclear reason - policies: {env.fasc_policies}, "
                                          f"last chancellor was not Hitler, phase: {env.phase}")
                                    reason = "other"
                            else:
                                print(f"WARNING: Fascist win with unclear reason - policies: {env.fasc_policies}, "
                                      f"no last chancellor, phase: {env.phase}")
                                reason = "other"
                            results["fasc_win_reasons"][reason] = results["fasc_win_reasons"].get(reason, 0) + 1
                    break  # Only count once per game

            if verbose and (game_num + 1) % 10 == 0:
                print(f"Completed {game_num + 1}/{num_games} games. lib wins: {results['lib_wins']}, fasc wins: {results['fasc_wins']}")

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

    print("Testing with instantiated agents:")
    # Create agent instances once (saves on initialization cost)
    agent_instances = [SimpleRandomAgent() for _ in range(5)]
    results = evaluate_agents(agent_instances, num_games=100, verbose=True, seed=42)

    print("\nResults:")
    print(f"Liberal win rate: {results['lib_win_rate']:.2%}")
    print(f"Fascist win rate: {results['fasc_win_rate']:.2%}")
    print(f"Average rewards: {results['avg_rewards']}")

    print("\n" + "="*50)
    print("\nTesting with lib/fasc agent lists:")
    # Create separate liberal and fascist agents
    lib_agents = [SimpleRandomAgent() for _ in range(3)]
    fasc_agents = [SimpleRandomAgent() for _ in range(2)]

    results = evaluate_agents(None, num_games=100, verbose=False, seed=42,
                            lib_agents=lib_agents, fasc_agents=fasc_agents)

    print("\nResults (with random role assignment):")
    print(f"Liberal win rate: {results['lib_win_rate']:.2%}")
    print(f"Fascist win rate: {results['fasc_win_rate']:.2%}")
