"""Train CFR agent for Silent Shitler."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.cfr import CFRTrainer
from utils.evaluation import run_games, print_results

# Wandb logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to console only")

# ============================================================================
# Config
# ============================================================================

USE_WANDB = True
WANDB_PROJECT = "silent-shitler"
WANDB_ENTITY = None

NUM_ITERATIONS = 500000  # 500k iterations
LOG_EVERY = 1000         # Log metrics every N iterations
EVAL_EVERY = 50000       # Full evaluation every N iterations
CHECKPOINT_EVERY = 50000 # Save checkpoint every N iterations
EVAL_GAMES = 500         # Games per evaluation


class CFRGameAgent:
    """Wrapper to use CFR agent with evaluation framework."""
    
    def __init__(self, cfr_agent, env=None):
        self.cfr_agent = cfr_agent
        self.env = env
    
    def get_action(self, obs, action_space):
        phase = self._infer_phase(obs, action_space)
        return self.cfr_agent.get_action(obs, action_space, phase)
    
    def _infer_phase(self, obs, action_space):
        """Infer game phase from observation."""
        if "nomination_mask" in obs:
            return "nomination"
        elif "execution_mask" in obs:
            return "execution"
        elif "cards" in obs:
            return "prez_cardsel" if len(obs["cards"]) == 4 else "chanc_cardsel"
        elif action_space.n == 4:
            return "prez_claim"
        elif action_space.n == 3:
            return "chanc_claim"
        else:
            return "voting"


def main():
    print("=" * 70)
    print("CFR TRAINING FOR SILENT SHITLER")
    print("=" * 70)
    
    # Setup
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    config = {
        "algorithm": "outcome_sampling_mccfr",
        "num_iterations": NUM_ITERATIONS,
        "log_every": LOG_EVERY,
        "eval_every": EVAL_EVERY,
        "eval_games": EVAL_GAMES,
    }
    
    # Print config
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    print()
    
    # Initialize wandb
    use_wandb = USE_WANDB and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=config,
            name=f"cfr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        print(f"Wandb logging enabled: {wandb.run.url}")
    else:
        print("Wandb logging disabled")
    print()
    
    # Create trainer
    trainer = CFRTrainer(checkpoint_dir=str(checkpoint_dir))
    
    start_time = time.time()
    last_log_time = start_time
    last_infosets = 0
    
    for iteration in range(1, NUM_ITERATIONS + 1):
        # Single iteration
        trainer.train(num_iterations=1, checkpoint_every=None, verbose=False)
        
        # Periodic logging
        if iteration % LOG_EVERY == 0:
            now = time.time()
            elapsed = now - start_time
            batch_time = now - last_log_time
            last_log_time = now
            
            num_infosets = len(trainer.cfr_agent.regret_sums)
            new_infosets = num_infosets - last_infosets
            last_infosets = num_infosets
            
            iters_per_sec = LOG_EVERY / batch_time
            
            print(f"[{iteration:>7}/{NUM_ITERATIONS}] "
                  f"infosets={num_infosets:>7} (+{new_infosets:>5}) | "
                  f"{iters_per_sec:.0f} it/s | "
                  f"elapsed={elapsed:.0f}s")
            
            if use_wandb:
                wandb.log({
                    "iteration": iteration,
                    "infosets": num_infosets,
                    "new_infosets": new_infosets,
                    "iters_per_sec": iters_per_sec,
                    "elapsed_seconds": elapsed,
                }, step=iteration)
        
        # Periodic evaluation
        if iteration % EVAL_EVERY == 0:
            print(f"\n--- Evaluation at iteration {iteration} ---")
            
            def agent_factory():
                return CFRGameAgent(trainer.cfr_agent)
            
            eval_start = time.time()
            results = run_games(
                agent_factory=agent_factory,
                num_games=EVAL_GAMES,
                seed=42,
                verbose=False
            )
            eval_time = time.time() - eval_start
            
            print_results(results)
            print(f"Evaluation took {eval_time:.1f}s")
            
            if use_wandb:
                wandb.log({
                    "eval/liberal_win_rate": results["liberal_win_rate"],
                    "eval/fascist_win_rate": results["fascist_win_rate"],
                    "eval/avg_game_length": results["avg_game_length"],
                    "eval/lib_5_policies": results["win_conditions"]["lib_5_policies"],
                    "eval/fasc_6_policies": results["win_conditions"]["fasc_6_policies"],
                    "eval/hitler_chancellor": results["win_conditions"]["hitler_chancellor"],
                    "eval/hitler_executed": results["win_conditions"]["hitler_executed"],
                }, step=iteration)
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"cfr_{iteration}_{timestamp}.json"
            
            results_with_meta = {
                "agent_type": "cfr_outcome_sampling",
                "iterations": iteration,
                "infosets": len(trainer.cfr_agent.regret_sums),
                "training_time_seconds": time.time() - start_time,
                "timestamp": timestamp,
                **results
            }
            
            with open(results_file, "w") as f:
                json.dump(results_with_meta, f, indent=2)
            print(f"Results saved to {results_file}\n")
        
        # Periodic checkpointing
        if iteration % CHECKPOINT_EVERY == 0:
            trainer.save_checkpoint()
    
    # Final save
    total_time = time.time() - start_time
    trainer.save_checkpoint()
    
    print("=" * 70)
    print(f"Training complete!")
    print(f"Total iterations: {NUM_ITERATIONS}")
    print(f"Total infosets: {len(trainer.cfr_agent.regret_sums)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
