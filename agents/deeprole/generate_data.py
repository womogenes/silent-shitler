import os
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

    trainer = BackwardsTrainer(num_workers=96)

    # Generate data for all game states using backwards training order
    # This will take significant time - recommend running on cluster

    n_samples = 10_000
    cfr_iters = 16
    cfr_delay = 15

    trainer.train_all_networks(
        samples_per_stage=n_samples,                   # Number of samples per (lib, fasc) state
        cfr_iterations=cfr_iters,                      # CFR iterations per sample
        cfr_delay=cfr_delay,                           # Averaging delay (paper default)
        save_path=f"trained_networks_{n_samples}_{cfr_iters}_{cfr_delay}.pkl"
    )

if __name__ == "__main__":
    # Guard to prevent re-execution in spawn mode
    main()
