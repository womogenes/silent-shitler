#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import numpy as np
from agents.deeprole.vector_cfr import VectorCFR
from agents.deeprole.game_state import create_game_at_state
from agents.deeprole.situation_sampler import AdvancedSituationSampler
from agents.deeprole.backwards_training import BackwardsTrainer

if __name__ == "__main__":
    print("RUNTIME ESTIMATION FOR 3-HOUR TRAINING")
    print("=" * 60)

    # Test a few representative states
    test_states = [
        (4, 5, "Near terminal - slowest"),
        (3, 3, "Mid-game - average"),
        (1, 1, "Early game - faster"),
    ]

    cfr_iterations = 75  # Proposed for 3-hour run
    cfr_delay = 25

    times = []

    for lib, fasc, desc in test_states:
        print(f"\nTesting ({lib}L, {fasc}F) - {desc}")

        # Sample setup
        sampler = AdvancedSituationSampler()
        president_idx, belief = sampler.sample_situation_with_constraints(lib, fasc)
        env = create_game_at_state(lib, fasc, president_idx, seed=42)
        cfr = VectorCFR()

        # Time one sample
        max_depth = 3 if (lib >= 4 or fasc >= 5) else 5

        start = time.time()
        values = cfr.solve_situation(
            env, belief,
            num_iterations=cfr_iterations,
            averaging_delay=cfr_delay,
            neural_nets=None,
            max_depth=max_depth
        )
        elapsed = time.time() - start

        times.append(elapsed)
        print(f"  Time per sample: {elapsed:.2f}s")
        print(f"  Max depth used: {max_depth}")

    # Calculate estimates
    avg_time = np.mean(times)
    max_time = np.max(times)

    print("\n" + "=" * 60)
    print("ESTIMATES FOR FULL TRAINING")
    print("=" * 60)

    trainer = BackwardsTrainer()
    num_states = len(trainer._get_ordered_game_parts())

    for samples_per_state in [100, 200, 300, 500]:
        total_samples = num_states * samples_per_state

        # Conservative estimate (using max time)
        conservative_time = total_samples * max_time
        # Optimistic estimate (using average)
        optimistic_time = total_samples * avg_time

        print(f"\nWith {samples_per_state} samples per state ({total_samples} total):")
        print(f"  Optimistic: {optimistic_time/3600:.1f} hours")
        print(f"  Conservative: {conservative_time/3600:.1f} hours")

        if optimistic_time < 3 * 3600:
            print(f"  ✓ Should fit in 3-hour window")
        elif conservative_time < 3 * 3600:
            print(f"  ⚠️ Might fit in 3-hour window")
        else:
            print(f"  ✗ Won't fit in 3-hour window")

    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR 3-HOUR RUN:")
    print("=" * 60)
    print(f"  CFR iterations: 75")
    print(f"  CFR delay: 25")
    print(f"  Samples per state: 200")
    print(f"  Number of workers: 4 (adjust to your CPU)")
    print(f"  Expected coverage: ~20-25 states out of 30")
    print()
    print("The resumable script will:")
    print("  1. Save after each state")
    print("  2. Auto-resume if interrupted")
    print("  3. Stop gracefully at 3-hour mark")
    print("  4. Allow you to continue later")