"""Sophisticated terminal value computation using CFR with sampled histories."""

import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
import copy

from .situation_sampler import AdvancedSituationSampler
from .vector_cfr import VectorCFR
from .game_state import create_game_at_state


class TerminalValueComputer:
    """Computes diverse terminal values using CFR on sampled game histories."""

    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.sampler = AdvancedSituationSampler()

    def generate_terminal_values(self, lib_policies, fasc_policies, n_samples):
        """Generate terminal values with strategic diversity through CFR.

        Instead of simple belief-weighted averages, this:
        1. Samples plausible game histories leading to terminal state
        2. Runs CFR from late-game position to compute strategic values
        3. Creates diversity through different histories and beliefs
        """
        print(f"  Generating {n_samples} sophisticated terminal samples...")

        if lib_policies >= 5:
            return self._generate_liberal_win_values(n_samples)
        else:  # fasc_policies >= 6
            return self._generate_fascist_win_values(n_samples)

    def _generate_liberal_win_values(self, n_samples):
        """Generate values for liberal win (5L) using CFR on sampled histories."""

        # Prepare args for parallel processing
        args_list = [(5, i) for i in range(n_samples)]

        print("  Sampling game histories and running CFR...")
        with Pool(self.num_workers) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(self._compute_liberal_win_sample, args_list),
                total=n_samples,
                desc="  Liberal wins",
                ncols=80
            ))

        # Convert to training data format
        training_data = []
        for president_idx, belief, values in results:
            president_oh = torch.zeros(5)
            president_oh[president_idx] = 1
            inp = torch.cat([president_oh, torch.tensor(belief, dtype=torch.float32)])
            tgt = torch.tensor(values, dtype=torch.float32)
            training_data.append((inp, tgt))

        return training_data

    def _generate_fascist_win_values(self, n_samples):
        """Generate values for fascist win (6F) using CFR on sampled histories."""

        # Prepare args for parallel processing
        args_list = [(6, i) for i in range(n_samples)]

        print("  Sampling game histories and running CFR...")
        with Pool(self.num_workers) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(self._compute_fascist_win_sample, args_list),
                total=n_samples,
                desc="  Fascist wins",
                ncols=80
            ))

        # Convert to training data format
        training_data = []
        for president_idx, belief, values in results:
            president_oh = torch.zeros(5)
            president_oh[president_idx] = 1
            inp = torch.cat([president_oh, torch.tensor(belief, dtype=torch.float32)])
            tgt = torch.tensor(values, dtype=torch.float32)
            training_data.append((inp, tgt))

        return training_data

    def _compute_liberal_win_sample(self, args):
        """Compute single liberal win sample using CFR on sampled history."""
        policies_enacted, seed = args
        np.random.seed(seed)

        # Sample diverse situation
        president_idx, belief = self.sampler.sample_situation_with_constraints(5, 0)

        # Sample a plausible game history leading to 5L
        history = self._sample_liberal_win_history()

        # Create game state at 4L (one round before terminal)
        # This gives CFR something to reason about
        env = create_game_at_state(4, history['fasc_before_win'], president_idx, seed)

        # Run short CFR to compute strategic values
        cfr = VectorCFR()
        values = cfr.solve_situation(
            env,
            belief,
            num_iterations=100,  # Fewer iterations since near terminal
            averaging_delay=20,
            neural_nets=None,
            max_depth=3  # Shallow depth for terminal evaluation
        )

        # Apply liberal win bonus to values based on role probabilities
        # This ensures liberals get positive final values
        for i, assignment in enumerate(self.sampler.manager.assignments):
            for player in range(5):
                if assignment[player] == 0:  # Liberal
                    values[player] += belief[i] * 0.5  # Bonus for winning
                else:  # Fascist/Hitler
                    values[player] -= belief[i] * 0.5  # Penalty for losing

        return president_idx, belief, values

    def _compute_fascist_win_sample(self, args):
        """Compute single fascist win sample using CFR on sampled history."""
        policies_enacted, seed = args
        np.random.seed(seed)

        # Sample diverse situation
        president_idx, belief = self.sampler.sample_situation_with_constraints(0, 6)

        # Sample a plausible game history leading to 6F
        history = self._sample_fascist_win_history()

        # Create game state at 5F (one round before terminal)
        env = create_game_at_state(history['lib_before_win'], 5, president_idx, seed)

        # Run short CFR to compute strategic values
        cfr = VectorCFR()
        values = cfr.solve_situation(
            env,
            belief,
            num_iterations=100,
            averaging_delay=20,
            neural_nets=None,
            max_depth=3
        )

        # Apply fascist win bonus to values based on role probabilities
        for i, assignment in enumerate(self.sampler.manager.assignments):
            for player in range(5):
                if assignment[player] > 0:  # Fascist or Hitler
                    values[player] += belief[i] * 0.5  # Bonus for winning
                else:  # Liberal
                    values[player] -= belief[i] * 0.5  # Penalty for losing

        return president_idx, belief, values

    def _sample_liberal_win_history(self):
        """Sample a plausible history leading to liberal win.

        Returns dict with:
        - governments: List of (president, chancellor) pairs that passed liberal
        - fasc_before_win: Number of fascist policies passed (0-2)
        - voting_patterns: How players voted on successful governments
        """
        # Randomly decide how many fascist policies were passed
        fasc_before_win = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])

        # Sample which players were in successful liberal governments
        governments = []
        for _ in range(5):  # Need 5 liberal policies
            president = np.random.randint(5)
            chancellor = np.random.choice([i for i in range(5) if i != president])
            governments.append((president, chancellor))

        # Sample voting patterns (some fascists vote yes to blend in)
        voting_patterns = []
        for gov in governments:
            votes = []
            for player in range(5):
                if player in gov:
                    # Government members usually vote yes
                    vote = np.random.choice([0, 1], p=[0.1, 0.9])
                else:
                    # Others vote more randomly
                    vote = np.random.choice([0, 1], p=[0.4, 0.6])
                votes.append(vote)
            voting_patterns.append(votes)

        return {
            'governments': governments,
            'fasc_before_win': fasc_before_win,
            'voting_patterns': voting_patterns
        }

    def _sample_fascist_win_history(self):
        """Sample a plausible history leading to fascist win.

        Returns dict with:
        - governments: List of (president, chancellor) pairs that passed fascist
        - lib_before_win: Number of liberal policies passed (0-4)
        - voting_patterns: How players voted
        """
        # Randomly decide how many liberal policies were passed
        lib_before_win = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.3, 0.1])

        # Sample which players were in successful fascist governments
        governments = []
        for _ in range(6):  # Need 6 fascist policies
            president = np.random.randint(5)
            chancellor = np.random.choice([i for i in range(5) if i != president])
            governments.append((president, chancellor))

        # Sample voting patterns
        voting_patterns = []
        for gov in governments:
            votes = []
            for player in range(5):
                # More variable voting for fascist governments
                if player in gov:
                    vote = np.random.choice([0, 1], p=[0.2, 0.8])
                else:
                    vote = np.random.choice([0, 1], p=[0.5, 0.5])
                votes.append(vote)
            voting_patterns.append(votes)

        return {
            'governments': governments,
            'lib_before_win': lib_before_win,
            'voting_patterns': voting_patterns
        }


class ImprovedBackwardsTrainer:
    """Backwards trainer with sophisticated terminal value computation."""

    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.sampler = AdvancedSituationSampler()
        self.terminal_computer = TerminalValueComputer(num_workers)
        self.networks = {}  # Will be NetworkEnsemble

    def generate_training_data(self, lib_policies, fasc_policies, n_samples,
                              cfr_iterations, cfr_delay):
        """Generate training data with sophisticated terminal evaluation."""

        print(f"  Generating {n_samples} training samples...")

        # Use sophisticated CFR-based computation for terminal states
        if lib_policies >= 5 or fasc_policies >= 6:
            return self.terminal_computer.generate_terminal_values(
                lib_policies, fasc_policies, n_samples
            )

        # Non-terminal states use regular CFR (same as before)
        args_list = [
            (lib_policies, fasc_policies, cfr_iterations, cfr_delay, i)
            for i in range(n_samples)
        ]

        print("  Running CFR for non-terminal positions...")
        with Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._generate_single_sample, args_list),
                total=n_samples,
                desc="  Non-terminal",
                ncols=80
            ))

        training_data = []
        for president_idx, belief, values in results:
            president_oh = torch.zeros(5)
            president_oh[president_idx] = 1
            inp = torch.cat([president_oh, torch.tensor(belief, dtype=torch.float32)])
            tgt = torch.tensor(values, dtype=torch.float32)
            training_data.append((inp, tgt))

        print(f"  Generated {len(training_data)} samples")
        return training_data

    def _generate_single_sample(self, args):
        """Generate single non-terminal sample (unchanged)."""
        lib_policies, fasc_policies, cfr_iterations, cfr_delay, seed = args
        np.random.seed(seed)

        president_idx, belief = self.sampler.sample_situation_with_constraints(
            lib_policies, fasc_policies
        )

        env = create_game_at_state(lib_policies, fasc_policies, president_idx, seed)
        cfr = VectorCFR()

        values = cfr.solve_situation(
            env,
            belief,
            num_iterations=cfr_iterations,
            averaging_delay=cfr_delay,
            neural_nets=self.networks,
            max_depth=5
        )

        return president_idx, belief, values