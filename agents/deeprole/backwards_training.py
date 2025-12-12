"""Backwards training procedure for DeepRole (Algorithm 3)."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from multiprocessing import Pool
import pickle

from .situation_sampler import AdvancedSituationSampler
from .networks import ValueNetwork, NetworkEnsemble
from .vector_cfr import VectorCFR
from .game_state import create_game_at_state, get_state_dependencies


class BackwardsTrainer:
    """Implements Algorithm 3: Backwards training for DeepRole."""

    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.sampler = AdvancedSituationSampler()
        self.networks = NetworkEnsemble()

    def train_all_networks(self, samples_per_stage=1000, cfr_iterations=100,
                          cfr_delay=30, save_path=None):
        """Train neural networks for all game stages using backwards training.

        Algorithm 3 from the paper.
        Starts from terminal states and works backwards, using trained
        networks from later stages as leaf evaluators.

        Args:
            samples_per_stage: Number of training samples per game stage
            cfr_iterations: CFR iterations per sample (paper uses 1500)
            cfr_delay: Iterations before averaging (paper uses 500)
            save_path: Where to save trained networks
        """
        # Get dependency-ordered list of game parts
        game_parts = self._get_ordered_game_parts()

        print(f"Training {len(game_parts)} networks using backwards training")
        print(f"Samples per stage: {samples_per_stage}")
        print(f"CFR iterations: {cfr_iterations}")
        print("-" * 60)

        # Train network for each game part
        for part_idx, (lib, fasc) in enumerate(game_parts):
            print(f"\n[{part_idx+1}/{len(game_parts)}] Training network for ({lib}L, {fasc}F)")

            # Generate training data
            training_data = self.generate_training_data(
                lib, fasc, samples_per_stage, cfr_iterations, cfr_delay
            )

            # Train neural network
            network = self.train_network(training_data, lib, fasc)

            # Add to ensemble for use in earlier stages
            self.networks.add_network(lib, fasc, network)

            # Save periodically
            if save_path and (part_idx + 1) % 5 == 0:
                self.networks.save(save_path)
                print(f"  Saved networks to {save_path}")

        # Final save
        if save_path:
            self.networks.save(save_path)
            print(f"\nTraining complete! Saved all networks to {save_path}")

        return self.networks

    def generate_training_data(self, lib_policies, fasc_policies, n_samples,
                              cfr_iterations, cfr_delay):
        """Generate training data for one game stage (Algorithm 3, GENERATEDATAPOINTS).

        Args:
            lib_policies: Number of liberal policies enacted
            fasc_policies: Number of fascist policies enacted
            n_samples: Number of samples to generate
            cfr_iterations: CFR iterations per sample
            cfr_delay: Iterations before averaging

        Returns:
            List of (input, target) pairs for neural network training
        """
        print(f"  Generating {n_samples} training samples...")

        # Terminal states don't need CFR
        if lib_policies >= 5 or fasc_policies >= 6:
            return self._generate_terminal_data(lib_policies, fasc_policies, n_samples)

        # For non-terminal states, use CFR with neural nets as leaf evaluators
        training_data = []

        # Use multiprocessing for parallel data generation
        args_list = [
            (lib_policies, fasc_policies, cfr_iterations, cfr_delay, i)
            for i in range(n_samples)
        ]

        with Pool(self.num_workers) as pool:
            results = pool.map(self._generate_single_sample, args_list)

        for president_idx, belief, values in results:
            # Prepare input: one-hot president + belief
            president_oh = torch.zeros(5)
            president_oh[president_idx] = 1
            input_tensor = torch.cat([
                president_oh,
                torch.tensor(belief, dtype=torch.float32)
            ])

            # Target is the value vector
            target_tensor = torch.tensor(values, dtype=torch.float32)

            training_data.append((input_tensor, target_tensor))

        print(f"  Generated {len(training_data)} samples")
        return training_data

    def _generate_single_sample(self, args):
        """Generate single training sample (for multiprocessing)."""
        lib_policies, fasc_policies, cfr_iterations, cfr_delay, seed = args
        np.random.seed(seed)

        # Sample game situation
        president_idx, belief = self.sampler.sample_situation_with_constraints(
            lib_policies, fasc_policies
        )

        # Create game at this state
        env = create_game_at_state(lib_policies, fasc_policies, president_idx, seed)

        # Solve with CFR using neural nets for deeper states
        cfr = VectorCFR()
        values = cfr.solve_situation(
            env,
            belief,
            num_iterations=cfr_iterations,
            averaging_delay=cfr_delay,
            neural_nets=self.networks.networks,  # Use already-trained networks
            max_depth=5  # Reasonable depth with neural net evaluation
        )

        return president_idx, belief, values

    def _generate_terminal_data(self, lib_policies, fasc_policies, n_samples):
        """Generate training data for terminal states (no CFR needed)."""
        training_data = []

        for _ in range(n_samples):
            # Sample situation
            president_idx, belief = self.sampler.sample_situation_with_constraints(
                lib_policies, fasc_policies
            )

            # Compute exact values for terminal state
            if lib_policies >= 5:
                # Liberal win
                values = np.zeros(5)
                for i, assignment in enumerate(self.sampler.manager.assignments):
                    for player in range(5):
                        if assignment[player] == 0:  # Liberal
                            values[player] += belief[i]
                        else:  # Fascist/Hitler
                            values[player] -= belief[i]
            else:  # fasc_policies >= 6
                # Fascist win
                values = np.zeros(5)
                for i, assignment in enumerate(self.sampler.manager.assignments):
                    for player in range(5):
                        if assignment[player] == 0:  # Liberal
                            values[player] -= belief[i]
                        else:  # Fascist/Hitler
                            values[player] += belief[i]

            # Prepare tensors
            president_oh = torch.zeros(5)
            president_oh[president_idx] = 1
            input_tensor = torch.cat([
                president_oh,
                torch.tensor(belief, dtype=torch.float32)
            ])
            target_tensor = torch.tensor(values, dtype=torch.float32)

            training_data.append((input_tensor, target_tensor))

        return training_data

    def train_network(self, training_data, lib_policies, fasc_policies):
        """Train a neural network on the generated data.

        Args:
            training_data: List of (input, target) pairs
            lib_policies: Number of liberal policies (for logging)
            fasc_policies: Number of fascist policies (for logging)

        Returns:
            Trained ValueNetwork
        """
        print(f"  Training neural network for ({lib_policies}L, {fasc_policies}F)...")

        network = ValueNetwork()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Split data into train/val
        n_train = int(0.9 * len(training_data))
        train_data = training_data[:n_train]
        val_data = training_data[n_train:]

        # Training loop (simplified - paper uses 3000 epochs)
        batch_size = min(32, len(train_data))
        n_epochs = min(100, 3000)  # Reduced for speed

        for epoch in range(n_epochs):
            # Shuffle training data
            np.random.shuffle(train_data)

            # Train in batches
            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                inputs = torch.stack([x[0] for x in batch])
                targets = torch.stack([x[1] for x in batch])

                # Forward pass
                president_indices = torch.argmax(inputs[:, :5], dim=1)
                beliefs = inputs[:, 5:]

                predicted_values = []
                for j in range(len(batch)):
                    values = network(president_indices[j], beliefs[j])
                    # Extract values for each player
                    player_values = []
                    for player_idx in range(5):
                        # Average over all assignments weighted by belief
                        weighted_value = (values[player_idx] * beliefs[j]).sum()
                        player_values.append(weighted_value)
                    predicted_values.append(torch.stack(player_values))

                predictions = torch.stack(predicted_values)

                # Compute loss
                loss = criterion(predictions, targets)
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            if epoch % 20 == 0 and val_data:
                val_loss = self._compute_validation_loss(network, val_data, criterion)
                print(f"    Epoch {epoch}: train_loss={total_loss/len(train_data):.4f}, "
                      f"val_loss={val_loss:.4f}")

        print(f"  Network trained for ({lib_policies}L, {fasc_policies}F)")
        return network

    def _compute_validation_loss(self, network, val_data, criterion):
        """Compute validation loss."""
        network.eval()
        total_loss = 0

        with torch.no_grad():
            for input_tensor, target in val_data:
                president_idx = torch.argmax(input_tensor[:5])
                belief = input_tensor[5:]

                values = network(president_idx, belief)
                player_values = []
                for player_idx in range(5):
                    weighted_value = (values[player_idx] * belief).sum()
                    player_values.append(weighted_value)
                prediction = torch.stack(player_values)

                loss = criterion(prediction, target)
                total_loss += loss.item()

        network.train()
        return total_loss / len(val_data)

    def _get_ordered_game_parts(self):
        """Get dependency-ordered list of game parts.

        Terminal states first, then working backwards based on what
        states can reach them.
        """
        parts = []

        # Terminal states first
        # Liberal wins
        for fasc in range(6):
            parts.append((5, fasc))

        # Fascist wins
        for lib in range(5):
            parts.append((lib, 6))

        # Near-terminal states (one policy away)
        for fasc in range(6):
            if (4, fasc) not in parts:
                parts.append((4, fasc))

        for lib in range(5):
            if (lib, 5) not in parts:
                parts.append((lib, 5))

        # Work backwards through remaining states
        for total_policies in range(8, -1, -1):
            for lib in range(min(5, total_policies + 1)):
                fasc = total_policies - lib
                if fasc <= 6 and (lib, fasc) not in parts:
                    parts.append((lib, fasc))

        return parts