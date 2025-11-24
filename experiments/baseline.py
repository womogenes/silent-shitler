"""
Baseline evaluation using random agents.

This script runs 10,000 games with random agents to establish a baseline
for comparing future learning algorithms (PPO, CFR, etc.).
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from agents.random_agent import RandomAgent
from utils.evaluation import run_games, print_results


def main():
    print("Running baseline evaluation with random agents...")

    # Run 10,000 games
    results = run_games(
        agent_factory=lambda: RandomAgent(),
        num_games=10000,
        seed=42,
        verbose=True
    )

    # Print results to console
    print_results(results)

    # Save results to JSON
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_{timestamp}.json"

    results_with_metadata = {
        "agent_type": "random",
        "timestamp": timestamp,
        "description": "Baseline evaluation with random agents",
        **results
    }

    with open(output_file, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
