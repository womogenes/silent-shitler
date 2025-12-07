"""
Evaluate trained liberal policy vs trained fascist policy.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from agents.ppo.ppo_agent import PPOAgent
from agents.ppo.observation import ObservationProcessor
from utils.evaluation import run_games, print_results


class TrainedVsTrainedAgent:
    """
    Agent that uses trained liberal policy for liberals,
    and trained fascist policy for fascists.
    """

    def __init__(self, liberal_model_path, fascist_model_path, device="cpu"):
        """
        Args:
            liberal_model_path: Path to trained liberal model
            fascist_model_path: Path to trained fascist model
            device: Device to run on
        """
        obs_processor = ObservationProcessor()
        obs_dim = obs_processor.obs_dim
        action_dim = 5  # Max actions: nomination/execution have 5 player options

        # Create two separate PPO agents
        self.liberal_agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            device=device,
        )

        self.fascist_agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            device=device,
        )

        # Load trained models
        print(f"Loading liberal model from: {liberal_model_path}")
        self.liberal_agent.load(liberal_model_path)

        print(f"Loading fascist model from: {fascist_model_path}")
        self.fascist_agent.load(fascist_model_path)

        print("✓ Both models loaded successfully!\n")

    def get_action(self, obs, action_space):
        """Get action from appropriate trained policy based on role."""
        # Infer role from observation (0=lib, 1=fasc, 2=hitler)
        role_encoded = obs["role"]

        # Determine which agent to use
        if role_encoded == 0:  # Liberal
            agent = self.liberal_agent
        else:  # Fascist or Hitler
            agent = self.fascist_agent

        # Get action (deterministic for eval)
        obs_array = agent.obs_processor.process(obs)
        n_valid = action_space.n
        action_mask = torch.zeros(6)
        action_mask[:n_valid] = 1.0

        action, _, _ = agent.get_action(obs_array, action_mask.numpy(), deterministic=True)

        return action


def main():
    print("=" * 70)
    print("TRAINED VS TRAINED EVALUATION")
    print("=" * 70)
    print("Liberals: Using trained liberal policy")
    print("Fascists: Using trained fascist policy")
    print("=" * 70)
    print()

    # Paths to trained models
    checkpoint_dir = Path(__file__).parent / "checkpoints_asymmetric"
    liberal_model = checkpoint_dir / "liberal" / "best_model.pt"
    fascist_model = checkpoint_dir / "fascist" / "best_model.pt"

    # Check if models exist
    if not liberal_model.exists():
        print(f"ERROR: Liberal model not found at {liberal_model}")
        print("Run train_ppo_asymmetric.py with TRAIN_TEAM='liberal' first!")
        return

    if not fascist_model.exists():
        print(f"ERROR: Fascist model not found at {fascist_model}")
        print("Run train_ppo_asymmetric.py with TRAIN_TEAM='fascist' first!")
        return

    # Load models once (not per game!)
    print("Loading trained models...")
    trained_agent = TrainedVsTrainedAgent(
        liberal_model_path=liberal_model,
        fascist_model_path=fascist_model,
    )

    # Create agent factory that returns the same agent
    def create_agent():
        return trained_agent

    # Run evaluation
    print("Running 1000 games with both trained models...")
    print("This will take a few minutes...\n")

    results = run_games(
        agent_factory=create_agent,
        num_games=1000,
        seed=42,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TRAINED VS TRAINED RESULTS")
    print("=" * 70)
    print()

    print_results(results)

    # Compare to baselines
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINES")
    print("=" * 70)

    print("\nRandom Baseline:")
    print(f"  Liberals:  28.6%")
    print(f"  Fascists:  71.4%")

    print("\nLibs Trained vs Random Fascists:")
    print(f"  Liberals:  32.5%")
    print(f"  Fascists:  67.5%")

    print("\nRandom Libs vs Trained Fascists:")
    print(f"  Liberals:  25.0%")
    print(f"  Fascists:  75.0%")

    print("\nBOTH TRAINED (this run):")
    print(f"  Liberals:  {results['liberal_win_rate']:.1%}")
    print(f"  Fascists:  {results['fascist_win_rate']:.1%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    lib_change = results['liberal_win_rate'] - 0.286
    print(f"\nLiberal performance vs random baseline: {lib_change:+.1%}")

    if results['liberal_win_rate'] > 0.325:
        print("✓ Liberals performing better than vs random fascists!")
        print("  → Liberal training helped even against trained fascists")
    elif results['liberal_win_rate'] > 0.286:
        print("~ Liberals improved from baseline but regressed from vs-random")
        print("  → Trained fascists countered some liberal strategies")
    else:
        print("✗ Liberals performing worse than baseline")
        print("  → Trained fascists are dominating")

    print()

    if results['fascist_win_rate'] > 0.75:
        print("✓ Fascists performing better than vs random liberals!")
        print("  → Fascist training helped even against trained liberals")
    elif results['fascist_win_rate'] > 0.714:
        print("~ Fascists improved from baseline but regressed from vs-random")
        print("  → Trained liberals countered some fascist strategies")
    else:
        print("~ Fascists performing similar to baseline")
        print("  → Training didn't help much against trained liberals")

    # Save results
    import json
    from datetime import datetime

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"trained_vs_trained_{timestamp}.json"

    results_with_metadata = {
        "agent_type": "trained_vs_trained",
        "timestamp": timestamp,
        "description": "Both teams using trained PPO policies",
        "liberal_model": str(liberal_model),
        "fascist_model": str(fascist_model),
        **results
    }

    with open(output_file, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
