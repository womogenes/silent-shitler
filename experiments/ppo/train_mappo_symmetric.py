"""
Symmetric MAPPO training script for Silent Shitler.

Both teams (liberals and fascists) train simultaneously with MAPPO.
This is the traditional MAPPO approach (self-play).
"""

import sys
from pathlib import Path
import torch
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from shitler_env.game import ShitlerEnv
from agents.ppo.symmetric_mappo_agent import SymmetricMAPPOAgent
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

# Logging
USE_WANDB = True  # Set to False to disable wandb logging
WANDB_PROJECT = "silent-shitler"
WANDB_ENTITY = None  # Set to your wandb username if needed

# MAPPO parameters (same as asymmetric for fair comparison)
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
N_STEPS_PER_ITERATION = 2048  # Collect this many steps per team before each update
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
# Helper: Symmetric MAPPO-based game player for evaluation
# ============================================================================

class SymmetricMAPPOGameAgent:
    """Wrapper to use symmetric MAPPO agent in evaluation framework."""

    def __init__(self, symmetric_agent):
        self.symmetric_agent = symmetric_agent

    def get_action(self, obs, action_space):
        """Get action from appropriate policy based on agent role."""
        # Infer role from observation (0=lib, 1=fasc, 2=hitler)
        role_encoded = obs["role"]
        role_map = {0: "lib", 1: "fasc", 2: "hitty"}
        agent_role = role_map[role_encoded]

        # Use appropriate MAPPO policy (deterministic for eval)
        obs_array = self.symmetric_agent.liberal_agent.obs_processor.process(obs)
        n_valid = action_space.n
        action_mask = torch.zeros(6)
        action_mask[:n_valid] = 1.0

        action, _, _ = self.symmetric_agent.get_action(
            obs_array, action_mask.numpy(), agent_role, deterministic=True
        )
        return action


# ============================================================================
# Main training loop
# ============================================================================

def main():
    print("=" * 70)
    print("Symmetric MAPPO Training for Silent Shitler")
    print("=" * 70)
    print(f"Algorithm: MAPPO (Multi-Agent PPO with Centralized Critic)")
    print(f"Training: BOTH TEAMS train simultaneously (self-play)")
    print(f"Device: {DEVICE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Hidden dims: {HIDDEN_DIMS}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Steps per iteration: {N_STEPS_PER_ITERATION} per team")
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
                "training_type": "symmetric",
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
            name=f"mappo_symmetric_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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

    # Create symmetric MAPPO agent
    agent = SymmetricMAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
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
    checkpoint_dir = Path(__file__).parent / "checkpoints_mappo_symmetric"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_time = time.time()
    best_liberal_win_rate = 0.0

    # Baselines for comparison
    baseline_lib_win_rate = 0.286
    baseline_fasc_win_rate = 0.714
    print(f"Baseline liberal win rate: {baseline_lib_win_rate:.1%}")
    print(f"Baseline fascist win rate: {baseline_fasc_win_rate:.1%}")
    print(f"Asymmetric MAPPO: 30.2% (liberal) / 76.7% (fascist)")
    print(f"Goal: See if symmetric training hits equilibrium!\n")

    for iteration in range(1, N_ITERATIONS + 1):
        iter_start_time = time.time()

        # Collect rollouts (both teams)
        print(f"[Iteration {iteration}/{N_ITERATIONS}] Collecting rollouts...")
        liberal_buffer, fascist_buffer = agent.collect_rollouts(env, N_STEPS_PER_ITERATION)

        # Train both policies
        print(f"[Iteration {iteration}/{N_ITERATIONS}] Training both teams with MAPPO...")
        train_stats = agent.train(liberal_buffer, fascist_buffer, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)

        iter_time = time.time() - iter_start_time

        # Print training stats for both teams
        print(f"[Iteration {iteration}/{N_ITERATIONS}] Training stats:")
        print(f"  LIBERAL:")
        print(f"    Policy loss:    {train_stats['liberal']['policy_loss']:.4f}")
        print(f"    Value loss:     {train_stats['liberal']['value_loss']:.4f}")
        print(f"    Entropy:        {train_stats['liberal']['entropy']:.4f}")
        print(f"    Approx KL:      {train_stats['liberal']['approx_kl']:.4f}")
        print(f"    Clip fraction:  {train_stats['liberal']['clip_fraction']:.4f}")
        print(f"  FASCIST:")
        print(f"    Policy loss:    {train_stats['fascist']['policy_loss']:.4f}")
        print(f"    Value loss:     {train_stats['fascist']['value_loss']:.4f}")
        print(f"    Entropy:        {train_stats['fascist']['entropy']:.4f}")
        print(f"    Approx KL:      {train_stats['fascist']['approx_kl']:.4f}")
        print(f"    Clip fraction:  {train_stats['fascist']['clip_fraction']:.4f}")
        print(f"  Time: {iter_time:.1f}s")
        print()

        # Log training metrics to wandb
        if use_wandb:
            wandb.log({
                "train/liberal/policy_loss": train_stats['liberal']['policy_loss'],
                "train/liberal/value_loss": train_stats['liberal']['value_loss'],
                "train/liberal/entropy": train_stats['liberal']['entropy'],
                "train/liberal/approx_kl": train_stats['liberal']['approx_kl'],
                "train/liberal/clip_fraction": train_stats['liberal']['clip_fraction'],
                "train/fascist/policy_loss": train_stats['fascist']['policy_loss'],
                "train/fascist/value_loss": train_stats['fascist']['value_loss'],
                "train/fascist/entropy": train_stats['fascist']['entropy'],
                "train/fascist/approx_kl": train_stats['fascist']['approx_kl'],
                "train/fascist/clip_fraction": train_stats['fascist']['clip_fraction'],
                "train/iteration_time": iter_time,
                "iteration": iteration,
            })

        # Evaluate
        if iteration % EVAL_FREQ == 0:
            print(f"[Iteration {iteration}/{N_ITERATIONS}] Evaluating...")

            # Create evaluation agent factory
            def create_eval_agent():
                return SymmetricMAPPOGameAgent(agent)

            results = run_games(
                agent_factory=create_eval_agent,
                num_games=EVAL_EPISODES,
                seed=42 + iteration,
                verbose=False,
            )

            print(f"[Iteration {iteration}/{N_ITERATIONS}] Evaluation results:")
            print(f"  Liberal win rate: {results['liberal_win_rate']:.2%}")
            print(f"  Fascist win rate: {results['fascist_win_rate']:.2%}")
            print(f"  Liberal improvement: {(results['liberal_win_rate'] - baseline_lib_win_rate)*100:+.1f}%")
            print(f"  Fascist improvement: {(results['fascist_win_rate'] - baseline_fasc_win_rate)*100:+.1f}%")
            print(f"  Avg game length:  {results['avg_game_length']:.1f}")
            print()

            # Log evaluation metrics to wandb
            if use_wandb:
                wandb.log({
                    "eval/liberal_win_rate": results['liberal_win_rate'],
                    "eval/fascist_win_rate": results['fascist_win_rate'],
                    "eval/liberal_improvement": results['liberal_win_rate'] - baseline_lib_win_rate,
                    "eval/fascist_improvement": results['fascist_win_rate'] - baseline_fasc_win_rate,
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

            # Save best model (based on liberal win rate for consistency)
            if results['liberal_win_rate'] > best_liberal_win_rate:
                best_liberal_win_rate = results['liberal_win_rate']
                best_lib_path = checkpoint_dir / "best_liberal.pt"
                best_fasc_path = checkpoint_dir / "best_fascist.pt"
                agent.save(best_lib_path, best_fasc_path)
                print(f"  New best models saved! Liberal win rate: {best_liberal_win_rate:.2%}")
                print()

                if use_wandb:
                    wandb.run.summary["best_liberal_win_rate"] = best_liberal_win_rate
                    wandb.run.summary["best_iteration"] = iteration

        # Save checkpoint
        if iteration % SAVE_FREQ == 0:
            checkpoint_lib_path = checkpoint_dir / f"checkpoint_{iteration}_liberal.pt"
            checkpoint_fasc_path = checkpoint_dir / f"checkpoint_{iteration}_fascist.pt"
            agent.save(checkpoint_lib_path, checkpoint_fasc_path)
            print(f"[Iteration {iteration}/{N_ITERATIONS}] Checkpoints saved")
            print()

    # Training complete
    total_time = time.time() - start_time
    print("=" * 70)
    print("Training complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best liberal win rate: {best_liberal_win_rate:.2%}")
    print("=" * 70)

    # Final evaluation
    print("\nRunning final evaluation (1000 games)...")

    def create_final_eval_agent():
        return SymmetricMAPPOGameAgent(agent)

    final_results = run_games(
        agent_factory=create_final_eval_agent,
        num_games=1000,
        seed=9999,
        verbose=True,
    )

    print_results(final_results)

    print(f"\nPerformance Comparison:")
    print(f"  Baseline:           28.6% / 71.4%")
    print(f"  Asymmetric PPO:     32.5% / 75.0%")
    print(f"  Asymmetric MAPPO:   30.2% / 76.7%")
    print(f"  Symmetric MAPPO:    {final_results['liberal_win_rate']:.1%} / {final_results['fascist_win_rate']:.1%}")

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            "final/liberal_win_rate": final_results['liberal_win_rate'],
            "final/fascist_win_rate": final_results['fascist_win_rate'],
            "final/liberal_improvement": final_results['liberal_win_rate'] - baseline_lib_win_rate,
            "final/fascist_improvement": final_results['fascist_win_rate'] - baseline_fasc_win_rate,
            "final/avg_game_length": final_results['avg_game_length'],
            "final/total_training_time_minutes": total_time / 60,
        })
        wandb.run.summary["final_liberal_win_rate"] = final_results['liberal_win_rate']
        wandb.run.summary["final_fascist_win_rate"] = final_results['fascist_win_rate']
        wandb.run.summary["total_training_time_minutes"] = total_time / 60

    # Save final models
    final_lib_path = checkpoint_dir / "final_liberal.pt"
    final_fasc_path = checkpoint_dir / "final_fascist.pt"
    agent.save(final_lib_path, final_fasc_path)
    print(f"\nFinal models saved to {checkpoint_dir}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Wandb run finished")


if __name__ == "__main__":
    main()
