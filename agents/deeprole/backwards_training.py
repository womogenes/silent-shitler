import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from multiprocessing import Pool, set_start_method, get_start_method
import pickle
from tqdm import tqdm

# Set spawn mode for CUDA compatibility
try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from .situation_sampler import AdvancedSituationSampler
from .networks import ValueNetwork, NetworkEnsemble
from .vector_cfr import VectorCFR
from .game_state import create_game_at_state, get_state_dependencies


class BackwardsTrainer:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.sampler = AdvancedSituationSampler()
        self.networks = NetworkEnsemble()

    def train_all_networks(self, samples_per_stage=1000, cfr_iterations=100,
                           cfr_delay=30, save_path=None):
        game_parts = self._get_ordered_game_parts()
        print(f"Training {len(game_parts)} networks using backwards training")

        for part_idx, (lib, fasc) in enumerate(game_parts):
            print(f"\n[{part_idx+1}/{len(game_parts)}] Training network for ({lib}L, {fasc}F)")

            training_data = self.generate_training_data(
                lib, fasc, samples_per_stage, cfr_iterations, cfr_delay
            )

            network = self.train_network(training_data, lib, fasc)
            self.networks.add_network(lib, fasc, network)

            if save_path and (part_idx + 1) % 5 == 0:
                self.networks.save(save_path)
                print(f"  Saved networks to {save_path}")

        if save_path:
            self.networks.save(save_path)
            print(f"\nTraining complete! Saved all networks to {save_path}")

        return self.networks

    def generate_training_data(self, lib_policies, fasc_policies, n_samples,
                               cfr_iterations, cfr_delay):
        print(f"  Generating {n_samples} training samples...")

        if lib_policies >= 5 or fasc_policies >= 6:
            return self._generate_terminal_data(lib_policies, fasc_policies, n_samples)

        args_list = [
            (lib_policies, fasc_policies, cfr_iterations, cfr_delay, i)
            for i in range(n_samples)
        ]

        with Pool(self.num_workers) as pool:
            results = pool.map(self._generate_single_sample, args_list)

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
        lib_policies, fasc_policies, cfr_iterations, cfr_delay, seed = args
        # Create more diverse seeds by combining with game state
        combined_seed = seed * 1000 + lib_policies * 100 + fasc_policies * 10
        np.random.seed(combined_seed)

        # Use diverse sampling with varying concentration based on sample index
        concentration = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0][seed % 6]
        president_idx, belief = self.sampler._sample_with_concentration(
            lib_policies, fasc_policies, concentration
        )

        env = create_game_at_state(lib_policies, fasc_policies, president_idx, seed)
        cfr = VectorCFR()

        values = cfr.solve_situation(
            env,
            belief,
            num_iterations=cfr_iterations,
            averaging_delay=cfr_delay,
            neural_nets=self.networks.networks,
            max_depth=5
        )

        return president_idx, belief, values

    def _generate_terminal_data(self, lib_policies, fasc_policies, n_samples):
        training_data = []
        liberal_win = lib_policies >= 5

        for _ in tqdm(range(n_samples), ncols=80):
            president_idx, belief = self.sampler.sample_situation_with_constraints(
                lib_policies, fasc_policies
            )

            values = np.zeros(5)
            for i, assignment in enumerate(self.sampler.manager.assignments):
                for p in range(5):
                    if assignment[p] == 0:
                        values[p] += belief[i] if liberal_win else -belief[i]
                    else:
                        values[p] -= belief[i] if liberal_win else belief[i]

            president_oh = torch.zeros(5)
            president_oh[president_idx] = 1
            inp = torch.cat([president_oh, torch.tensor(belief, dtype=torch.float32)])
            tgt = torch.tensor(values, dtype=torch.float32)
            training_data.append((inp, tgt))

        return training_data

    def train_network(self, training_data, lib_policies, fasc_policies):
        print(f"  Training neural network for ({lib_policies}L, {fasc_policies}F)...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if this is a terminal state
        is_terminal = lib_policies >= 5 or fasc_policies >= 6

        network = ValueNetwork().to(device)

        # Use different settings for terminal vs non-terminal states
        if is_terminal:
            # Terminal states: less regularization, higher learning rate
            optimizer = torch.optim.Adam(network.parameters(), lr=0.01, weight_decay=1e-5)
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)

        criterion = nn.MSELoss()

        n_train = int(0.9 * len(training_data))
        train_data = training_data[:n_train]
        val_data = training_data[n_train:]

        batch_size = min(128, len(train_data))
        n_epochs = 100

        for epoch in tqdm(range(n_epochs), ncols=80, desc="Epoch"):
            network.train()
            np.random.shuffle(train_data)

            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]

                inputs = torch.stack([x[0] for x in batch]).to(device)
                targets = torch.stack([x[1] for x in batch]).to(device)

                president_indices = torch.argmax(inputs[:, :5], dim=1)
                beliefs = inputs[:, 5:]

                values = network(president_indices, beliefs)

                # Values shape: [batch, 5, 20] - value for each player under each assignment
                # beliefs shape: [batch, 20] - probability of each assignment
                # We need expected value per player: sum over assignments
                predictions = (values * beliefs.unsqueeze(1)).sum(dim=2)  # [batch, 5]

                loss = criterion(predictions, targets)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0 and val_data:
                val_loss = self._compute_validation_loss(network, val_data, criterion, device)
                print(f"    Epoch {epoch}: train_loss={total_loss/len(train_data):.4f}, val_loss={val_loss:.4f}")

        print(f"  Network trained for ({lib_policies}L, {fasc_policies}F)")
        return network

    def _compute_validation_loss(self, network, val_data, criterion, device):
        network.eval()
        losses = []

        with torch.no_grad():
            for inp, tgt in val_data:
                inp = inp.to(device)
                tgt = tgt.to(device)

                pidx = torch.argmax(inp[:5]).unsqueeze(0)
                belief = inp[5:].unsqueeze(0)

                values = network(pidx, belief)
                # values shape: [1, 5, 20], belief shape: [1, 20]
                pred = (values * belief.unsqueeze(1)).sum(dim=2).squeeze()  # [5]

                losses.append(criterion(pred, tgt).item())

        network.train()
        return sum(losses) / len(losses)

    def _get_ordered_game_parts(self):
        """Get game states in backwards training order.

        Terminal states (5L or 6F) are trained first, then work backwards.
        """
        parts = []
        # Terminal states - liberal wins (5L, 0-5F)
        for fasc in range(6):
            parts.append((5, fasc))
        # Terminal states - fascist wins (0-4L, 6F)
        for lib in range(5):
            parts.append((lib, 6))

        # Non-terminal states in reverse order (high to low total policies)
        for total in range(9, -1, -1):  # 9 down to 0 total policies
            for lib in range(min(5, total + 1)):
                fasc = total - lib
                if fasc <= 5 and (lib, fasc) not in parts:  # Only up to 5F for non-terminal
                    parts.append((lib, fasc))
        return parts
