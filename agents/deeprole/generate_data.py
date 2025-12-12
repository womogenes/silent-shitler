#!/usr/bin/env python3
"""Generate training data for DeepRole using sophisticated CFR evaluation.

This script generates data for all game states using backwards training order.
Terminal states now use CFR with sampled game histories for diversity.
Non-terminal states use standard CFR with neural network leaf evaluation.

Run with --quick flag for testing with reduced parameters.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import time

from deeprole.backwards_training import BackwardsTrainer

def main():
    parser = argparse.ArgumentParser(description="Generate DeepRole training data")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Samples per game state (default: 10000)")
    parser.add_argument("--cfr-iterations", type=int, default=1500,
                       help="CFR iterations for non-terminal states (default: 1500)")
    parser.add_argument("--cfr-delay", type=int, default=500,
                       help="CFR averaging delay (default: 500)")
    parser.add_argument("--workers", type=int, default=32,
                       help="Parallel workers (default: 32)")
    parser.add_argument("--output", type=str, default="trained_networks.pkl",
                       help="Output file (default: trained_networks.pkl)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode with minimal parameters")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Quick mode for testing
    if args.quick:
        print("QUICK TEST MODE - reduced parameters")
        args.samples = 20
        args.cfr_iterations = 50
        args.cfr_delay = 10
        args.workers = 4

    print("="*80)
    print("DEEPROLE TRAINING DATA GENERATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Samples per state: {args.samples}")
    print(f"  CFR iterations: {args.cfr_iterations}")
    print(f"  CFR delay: {args.cfr_delay}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")
    print(f"  Device: {'CUDA - ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print()

    print("Note: Terminal states use sophisticated CFR evaluation with game history")
    print("      sampling. This produces diverse, strategic training data but is")
    print("      computationally intensive (~10-20 samples/second).")
    print()

    start_time = time.time()

    trainer = BackwardsTrainer(num_workers=args.workers)

    # Generate data for all game states using backwards training order
    # This ensures later stages can use trained networks as leaf evaluators
    trainer.train_all_networks(
        samples_per_stage=args.samples,
        cfr_iterations=args.cfr_iterations,
        cfr_delay=args.cfr_delay,
        save_path=args.output
    )

    elapsed = time.time() - start_time
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60

    print()
    print("="*80)
    print(f"COMPLETE - Training finished in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Networks saved to: {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
