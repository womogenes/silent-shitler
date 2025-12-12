#!/usr/bin/env python3
"""Main training script for DeepRole networks.

Run this on your GPU cluster to train the complete set of neural networks.
Adjust parameters based on available compute resources.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import time

from backwards_training import BackwardsTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DeepRole networks")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Training samples per game state (default: 10000)")
    parser.add_argument("--cfr-iterations", type=int, default=1500,
                       help="CFR iterations per sample (default: 1500)")
    parser.add_argument("--cfr-delay", type=int, default=500,
                       help="CFR averaging delay (default: 500)")
    parser.add_argument("--workers", type=int, default=32,
                       help="Number of parallel workers (default: 32)")
    parser.add_argument("--output", type=str, default="trained_networks.pkl",
                       help="Output path for trained networks")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true",
                       help="Quick training with reduced parameters for testing")
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Quick mode for testing
    if args.quick:
        print("Running in quick mode with reduced parameters")
        args.samples = 100
        args.cfr_iterations = 50
        args.cfr_delay = 10

    print("DeepRole Training Configuration:")
    print(f"  Samples per stage: {args.samples}")
    print(f"  CFR iterations: {args.cfr_iterations}")
    print(f"  CFR delay: {args.cfr_delay}")
    print(f"  Workers: {args.workers}")
    print(f"  Output path: {args.output}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    start_time = time.time()

    # Create trainer
    trainer = BackwardsTrainer(num_workers=args.workers)

    # Run training
    trainer.train_all_networks(
        samples_per_stage=args.samples,
        cfr_iterations=args.cfr_iterations,
        cfr_delay=args.cfr_delay,
        save_path=args.output
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\nTraining completed in {hours}h {minutes}m")
    print(f"Networks saved to {args.output}")


if __name__ == "__main__":
    main()