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

    trainer = BackwardsTrainer(num_workers=1)  # Adjust based on CPU cores

    # Generate data for all game states using backwards training order
    # This will take significant time - recommend running on cluster
    trainer.train_all_networks(
        samples_per_stage=10000,  # Number of samples per (lib, fasc) state
        cfr_iterations=1500,      # CFR iterations per sample (paper default)
        cfr_delay=500,            # Averaging delay (paper default)
        save_path="trained_networks.pkl"
    )

if __name__ == "__main__":
    # Guard to prevent re-execution in spawn mode
    main()
