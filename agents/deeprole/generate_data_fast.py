#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

def main():
    # Only import and set up multiprocessing in main
    from deeprole.backwards_training import BackwardsTrainer

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    trainer = BackwardsTrainer(num_workers=2)  # Use 2 workers

    print("FAST TESTING VERSION - reduced parameters for quick iteration")
    print("="*60)
    print("Changes from paper defaults:")
    print("  - 100 samples instead of 10000 (100x faster)")
    print("  - 50 CFR iterations instead of 1500 (30x faster)")
    print("  - Total speedup: ~3000x")
    print()

    # Generate data for all game states using backwards training order
    # REDUCED PARAMETERS FOR TESTING
    trainer.train_all_networks(
        samples_per_stage=100,      # 100x fewer samples for testing
        cfr_iterations=50,           # 30x fewer iterations for testing
        cfr_delay=20,                # Proportionally reduced
        save_path="trained_networks_test.pkl"
    )

if __name__ == "__main__":
    # Guard to prevent re-execution in spawn mode
    main()