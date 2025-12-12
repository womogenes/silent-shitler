#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pickle
import os
import time
from datetime import datetime

def main():
    from deeprole.backwards_training import BackwardsTrainer

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration for ~3 hour run
    NUM_WORKERS = 4  # Adjust based on your CPU cores
    SAMPLES_PER_STATE = 200  # Reasonable sample count
    CFR_ITERATIONS = 75  # Balanced speed/quality
    CFR_DELAY = 25

    # File paths for resumable training
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PROGRESS_FILE = CHECKPOINT_DIR / "training_progress.pkl"
    NETWORKS_FILE = CHECKPOINT_DIR / "networks_partial.pkl"
    FINAL_FILE = "trained_networks_3hr.pkl"

    print("=" * 60)
    print("RESUMABLE TRAINING - 3 HOUR CONFIGURATION")
    print("=" * 60)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Samples per state: {SAMPLES_PER_STATE}")
    print(f"CFR iterations: {CFR_ITERATIONS}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print()

    # Load previous progress if it exists
    completed_states = set()
    start_time = time.time()

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
            completed_states = checkpoint['completed_states']
            elapsed_before = checkpoint.get('total_elapsed', 0)
            print(f"RESUMING from previous run")
            print(f"  Already completed: {len(completed_states)} states")
            print(f"  Previous time: {elapsed_before/60:.1f} minutes")
            print()
    else:
        elapsed_before = 0

    trainer = BackwardsTrainer(num_workers=NUM_WORKERS)

    # Load existing networks if available
    if NETWORKS_FILE.exists():
        trainer.networks.load(NETWORKS_FILE)
        print(f"Loaded existing networks from {NETWORKS_FILE}")

    # Get all states to train
    all_states = trainer._get_ordered_game_parts()
    remaining_states = [(lib, fasc) for lib, fasc in all_states
                        if (lib, fasc) not in completed_states]

    print(f"Total states: {len(all_states)}")
    print(f"Remaining states: {len(remaining_states)}")
    print(f"Time budget: 3 hours")
    print()

    # Train remaining states
    for idx, (lib, fasc) in enumerate(remaining_states):
        state_start = time.time()
        elapsed_total = (time.time() - start_time) + elapsed_before

        # Check time limit (leave 5 min buffer for final save)
        if elapsed_total > 3 * 3600 - 300:
            print(f"\nApproaching 3-hour limit, stopping gracefully...")
            break

        print(f"\n[{len(completed_states) + 1}/{len(all_states)}] "
              f"Training ({lib}L, {fasc}F)")
        print(f"  Elapsed: {elapsed_total/60:.1f} min, "
              f"Remaining time: {(3*3600 - elapsed_total)/60:.1f} min")

        try:
            # Generate training data
            training_data = trainer.generate_training_data(
                lib, fasc, SAMPLES_PER_STATE, CFR_ITERATIONS, CFR_DELAY
            )

            # Train network
            network = trainer.train_network(training_data, lib, fasc)
            trainer.networks.add_network(lib, fasc, network)

            # Mark as completed
            completed_states.add((lib, fasc))

            # Save checkpoint after each state
            checkpoint = {
                'completed_states': completed_states,
                'total_elapsed': (time.time() - start_time) + elapsed_before,
                'last_update': datetime.now().isoformat(),
                'config': {
                    'samples': SAMPLES_PER_STATE,
                    'cfr_iterations': CFR_ITERATIONS,
                    'cfr_delay': CFR_DELAY,
                    'num_workers': NUM_WORKERS
                }
            }
            with open(PROGRESS_FILE, 'wb') as f:
                pickle.dump(checkpoint, f)

            # Save networks
            trainer.networks.save(NETWORKS_FILE)

            state_time = time.time() - state_start
            print(f"  ✓ Completed in {state_time:.1f}s")
            print(f"  Checkpoint saved")

        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user")
            print(f"Progress saved - can resume later")
            break
        except Exception as e:
            print(f"\nError training ({lib}L, {fasc}F): {e}")
            print(f"Skipping this state and continuing...")
            continue

    # Final save
    elapsed_total = (time.time() - start_time) + elapsed_before
    print("\n" + "=" * 60)
    print(f"TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print(f"States completed: {len(completed_states)}/{len(all_states)}")

    if completed_states:
        trainer.networks.save(FINAL_FILE)
        print(f"Networks saved to: {FINAL_FILE}")

        # Print completed states
        print("\nCompleted states:")
        for lib, fasc in sorted(completed_states):
            print(f"  ({lib}L, {fasc}F)")

    if len(completed_states) < len(all_states):
        print(f"\nTo resume training, run this script again.")
        print(f"It will automatically continue from where it left off.")
    else:
        print(f"\nAll states trained successfully!")

if __name__ == "__main__":
    main()
