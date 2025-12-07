"""
Asymmetric MAPPO training script for Silent Shitler.

Trains one team (liberals or fascists) with MAPPO while the other team uses random policy.
MAPPO uses a centralized critic (sees global state) for better value estimation.
"""

import sys
from pathlib import Path
import torch
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from shitler_env.game import ShitlerEnv
from agents.ppo.asymmetric_mappo_agent import AsymmetricMAPPOAgent
from agents.ppo.observation import ObservationProcessor
from utils.evaluation import run_games, print_results

# Wandb logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to console only")


# ============================================================================
# Hyperparameters
# ============================================================================

# Asymmetric training settings
TRAIN_TEAM = "fascist"  # "liberal" or "fascist" - which team to train

# Logging
USE_WANDB = True  # Set to False to disable wandb logging
WANDB_PROJECT = "silent-shitler"
WANDB_ENTITY = None  # Set to your wandb username if needed

# MAPPO parameters (same as PPO for fair comparison)
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

# Network architecture
HIDDEN_DIMS = [128, 128]

# Training parameters
N_ITERATIONS = 1000
N_STEPS_PER_ITERATION = 2048  # Collect this many steps before each update
N_EPOCHS = 10  # Number of PPO epochs per iteration
BATCH_SIZE = 64

# Evaluation
EVAL_FREQ = 10  # Evaluate every N iterations
EVAL_EPISODES = 100  # Number of episodes for evaluation

# Checkpointing
SAVE_FREQ = 50  # Save checkpoint every N iterations

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Helper: Asymmetric MAPPO-based game player for evaluation
# ============================================================================

class AsymmetricMAPPOGameAgent:
    """Wrapper to use asymmetric MAPPO agent in evaluation framework."""

    def __init__(self, asymmetric_agent):
        self.asymmetric_agent = asymmetric_agent

    def get_action(self, obs, action_space):
        """Get action from appropriate policy based on agent role."""
        # Infer role from observation (0=lib, 1=fasc, 2=hitler)
        role_encoded = obs["role"]
        role_map = {0: "lib", 1: "fasc", 2: "hitty"}
        agent_role = role_map[role_encoded]

        # Check if this is trained team
        is_trained = self.asymmetric_agent._is_trained_team(agent_role)

        if is_trained:
            # Use MAPPO policy (deterministic for eval)
            obs_array = self.asymmetric_agent.mappo_agent.obs_processor.process(obs)
            n_valid = action_space.n
            action_mask = torch.zeros(6)
            action_mask[:n_valid] = 1.0

            action, _, _ = self.asymmetric_agent.get_action(
                obs_array, action_mask.numpy(), deterministic=True
            )
            return action
        else:
            # Use random policy
            return self.asymmetric_agent.random_agent.get_action(obs, action_space)


# ============================================================================
# Main training loop
# ============================================================================

def main():
    print("=" * 70)
    print("Asymmetric MAPPO Training for Silent Shitler")
    print("=" * 70)
    print(f"Algorithm: MAPPO (Multi-Agent PPO with Centralized Critic)")
    print(f"Training team: {TRAIN_TEAM.upper()}")
    print(f"Other team: RANDOM (frozen)")
    print(f"Device: {DEVICE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Hidden dims: {HIDDEN_DIMS}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Steps per iteration: {N_STEPS_PER_ITERATION}")
    print("=" * 70)
    print()

    # Initialize wandb
    use_wandb = USE_WANDB and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "algorithm": "MAPPO",
                "train_team": TRAIN_TEAM,
                "training_type": "asymmetric",
                "learning_rate": LEARNING_RATE,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_epsilon": CLIP_EPSILON,
                "value_coef": VALUE_COEF,
                "entropy_coef": ENTROPY_COEF,
                "max_grad_norm": MAX_GRAD_NORM,
                "hidden_dims": HIDDEN_DIMS,
                "n_iterations": N_ITERATIONS,
                "n_steps_per_iteration": N_STEPS_PER_ITERATION,
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
            },
            name=f"mappo_asymmetric_{TRAIN_TEAM}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        print(f"Wandb logging enabled: {wandb.run.url}")
        print()
    else:
        print("Wandb logging disabled")
        print()

    # Create environment
    env = ShitlerEnv()

    # Get observation and action dimensions
    obs_processor = ObservationProcessor()
    obs_dim = obs_processor.obs_dim
    action_dim = 5  # Max actions: nomination/execution have 5 player options

    print(f"Local observation dim: {obs_dim}")
    print(f"Global state dim: {obs_dim * 5} (5 agents)")
    print(f"Action dim: {action_dim}")
    print()

    # Create asymmetric MAPPO agent
    agent = AsymmetricMAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        train_team=TRAIN_TEAM,
        hidden_dims=HIDDEN_DIMS,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        device=DEVICE,
    )

    # Create checkpoint directory
    checkpoint_dir = Path(__file__).parent / "checkpoints_mappo_asymmetric" / TRAIN_TEAM
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_time = time.time()
    best_win_rate = 0.0

    # Baseline for comparison
    baseline_win_rate = 0.286 if TRAIN_TEAM == "liberal" else 0.714
    print(f"Baseline {TRAIN_TEAM} win rate: {baseline_win_rate:.1%}")
    print(f"PPO achieved: 34.5% (liberal) / 75.0% (fascist)")
    print(f"Goal: Match or beat PPO with MAPPO!\n")

    for iteration in range(1, N_ITERATIONS + 1):
        iter_start_time = time.time()

        # Collect rollouts
        print(f"[Iteration {iteration}/{N_ITERATIONS}] Collecting rollouts...")
        rollout_buffer = agent.collect_rollouts(env, N_STEPS_PER_ITERATION)

        # Train policy
        print(f"[Iteration {iteration}/{N_ITERATIONS}] Training {TRAIN_TEAM} policy with MAPPO...")
        train_stats = agent.train(rollout_buffer, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)

        iter_time = time.time() - iter_start_time

        # Print training stats
        print(f"[Iteration {iteration}/{N_ITERATIONS}] Training stats:")
        print(f"  Policy loss:    {train_stats['policy_loss']:.4f}")
        print(f"  Value loss:     {train_stats['value_loss']:.4f}")
        print(f"  Entropy:        {train_stats['entropy']:.4f}")
        print(f"  Approx KL:      {train_stats['approx_kl']:.4f}")
        print(f"  Clip fraction:  {train_stats['clip_fraction']:.4f}")
        print(f"  Time: {iter_time:.1f}s")
        print()

        # Log training metrics to wandb
        if use_wandb:
            wandb.log({
                "train/policy_loss": train_stats['policy_loss'],
                "train/value_loss": train_stats['value_loss'],
                "train/entropy": train_stats['entropy'],
                "train/approx_kl": train_stats['approx_kl'],
                "train/clip_fraction": train_stats['clip_fraction'],
                "train/iteration_time": iter_time,
                "iteration": iteration,
            })

        # Evaluate
        if iteration % EVAL_FREQ == 0:
            print(f"[Iteration {iteration}/{N_ITERATIONS}] Evaluating...")

            # Create evaluation agent factory
            def create_eval_agent():
                return AsymmetricMAPPOGameAgent(agent)

            results = run_games(
                agent_factory=create_eval_agent,
                num_games=EVAL_EPISODES,
                seed=42 + iteration,
                verbose=False,
            )

            # Get the trained team's win rate
            trained_win_rate = results['liberal_win_rate'] if TRAIN_TEAM == "liberal" else results['fascist_win_rate']

            print(f"[Iteration {iteration}/{N_ITERATIONS}] Evaluation results:")
            print(f"  Liberal win rate: {results['liberal_win_rate']:.2%}")
            print(f"  Fascist win rate: {results['fascist_win_rate']:.2%}")
            print(f"  {TRAIN_TEAM.capitalize()} win rate: {trained_win_rate:.2%} (baseline: {baseline_win_rate:.1%})")
            print(f"  Improvement: {(trained_win_rate - baseline_win_rate)*100:+.1f}%")
            print(f"  Avg game length:  {results['avg_game_length']:.1f}")
            print()

            # Log evaluation metrics to wandb
            if use_wandb:
                wandb.log({
                    "eval/liberal_win_rate": results['liberal_win_rate'],
                    "eval/fascist_win_rate": results['fascist_win_rate'],
                    "eval/trained_team_win_rate": trained_win_rate,
                    "eval/improvement_over_baseline": trained_win_rate - baseline_win_rate,
                    "eval/liberal_avg_reward": results['liberal_avg_reward'],
                    "eval/fascist_avg_reward": results['fascist_avg_reward'],
                    "eval/avg_game_length": results['avg_game_length'],
                    "eval/game_length_std": results['game_length_std'],
                    "eval/lib_5_policies": results['win_conditions']['lib_5_policies'],
                    "eval/fasc_6_policies": results['win_conditions']['fasc_6_policies'],
                    "eval/hitler_chancellor": results['win_conditions']['hitler_chancellor'],
                    "eval/hitler_executed": results['win_conditions']['hitler_executed'],
                    "iteration": iteration,
                })

            # Save best model
            if trained_win_rate > best_win_rate:
                best_win_rate = trained_win_rate
                best_path = checkpoint_dir / "best_model.pt"
                agent.save(best_path)
                print(f"  New best model saved! {TRAIN_TEAM.capitalize()} win rate: {best_win_rate:.2%}")
                print()

                if use_wandb:
                    wandb.run.summary[f"best_{TRAIN_TEAM}_win_rate"] = best_win_rate
                    wandb.run.summary["best_iteration"] = iteration

        # Save checkpoint
        if iteration % SAVE_FREQ == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
            agent.save(checkpoint_path)
            print(f"[Iteration {iteration}/{N_ITERATIONS}] Checkpoint saved to {checkpoint_path}")
            print()

    # Training complete
    total_time = time.time() - start_time
    print("=" * 70)
    print("Training complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best {TRAIN_TEAM} win rate: {best_win_rate:.2%}")
    print(f"Improvement over baseline: {(best_win_rate - baseline_win_rate)*100:+.1f}%")
    print("=" * 70)

    # Final evaluation
    print("\nRunning final evaluation (1000 games)...")

    def create_final_eval_agent():
        return AsymmetricMAPPOGameAgent(agent)

    final_results = run_games(
        agent_factory=create_final_eval_agent,
        num_games=1000,
        seed=9999,
        verbose=True,
    )

    print_results(final_results)

    final_trained_win_rate = final_results['liberal_win_rate'] if TRAIN_TEAM == "liberal" else final_results['fascist_win_rate']

    print(f"\n{TRAIN_TEAM.capitalize()} Performance Comparison:")
    print(f"  Baseline (random):  {baseline_win_rate:.1%}")
    print(f"  PPO:                {'34.5%' if TRAIN_TEAM == 'liberal' else '75.0%'}")
    print(f"  MAPPO (this run):   {final_trained_win_rate:.1%}")

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            "final/liberal_win_rate": final_results['liberal_win_rate'],
            "final/fascist_win_rate": final_results['fascist_win_rate'],
            "final/trained_team_win_rate": final_trained_win_rate,
            "final/improvement_over_baseline": final_trained_win_rate - baseline_win_rate,
            "final/avg_game_length": final_results['avg_game_length'],
            "final/total_training_time_minutes": total_time / 60,
        })
        wandb.run.summary[f"final_{TRAIN_TEAM}_win_rate"] = final_trained_win_rate
        wandb.run.summary["final_improvement"] = final_trained_win_rate - baseline_win_rate
        wandb.run.summary["total_training_time_minutes"] = total_time / 60

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    agent.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Wandb run finished")


if __name__ == "__main__":
    main()
