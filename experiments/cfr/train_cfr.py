"""Train CFR agent for Silent Shitler."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.cfr import CFRTrainer
from utils.evaluation import run_games, print_results


class CFRGameAgent:
    """Wrapper to use CFR agent with evaluation framework."""
    
    def __init__(self, cfr_agent, env=None):
        self.cfr_agent = cfr_agent
        self.env = env
    
    def get_action(self, obs, action_space):
        # Need to get phase from somewhere - infer from obs
        phase = self._infer_phase(obs, action_space)
        return self.cfr_agent.get_action(obs, action_space, phase)
    
    def _infer_phase(self, obs, action_space):
        """Infer game phase from observation."""
        if "nomination_mask" in obs:
            return "nomination"
        elif "execution_mask" in obs:
            return "execution"
        elif "cards" in obs:
            if len(obs["cards"]) == 4:
                return "prez_cardsel"
            else:
                return "chanc_cardsel"
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
    print()
    
    # Config
    num_iterations = 100000
    eval_every = 10000
    checkpoint_every = 10000
    eval_games = 200
    
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    trainer = CFRTrainer(checkpoint_dir=str(checkpoint_dir))
    
    print(f"Training for {num_iterations} iterations")
    print(f"Evaluating every {eval_every} iterations")
    print(f"Checkpointing every {checkpoint_every} iterations")
    print()
    
    start_time = time.time()
    
    for batch in range(num_iterations // eval_every):
        batch_start = time.time()
        
        # Train
        trainer.train(
            num_iterations=eval_every,
            checkpoint_every=checkpoint_every,
            verbose=True
        )
        
        batch_time = time.time() - batch_start
        total_time = time.time() - start_time
        
        print(f"\n--- Batch {batch + 1} complete ---")
        print(f"Iterations: {trainer.iterations}")
        print(f"Infosets: {len(trainer.cfr_agent.regret_sums)}")
        print(f"Batch time: {batch_time:.1f}s")
        print(f"Total time: {total_time:.1f}s")
        
        # Evaluate
        print(f"\nEvaluating over {eval_games} games...")
        
        def agent_factory():
            return CFRGameAgent(trainer.cfr_agent)
        
        results = run_games(
            agent_factory=agent_factory,
            num_games=eval_games,
            seed=42,
            verbose=False
        )
        
        print_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"cfr_{trainer.iterations}_{timestamp}.json"
        
        results_with_meta = {
            "agent_type": "cfr_outcome_sampling",
            "iterations": trainer.iterations,
            "infosets": len(trainer.cfr_agent.regret_sums),
            "training_time_seconds": total_time,
            **results
        }
        
        with open(results_file, "w") as f:
            json.dump(results_with_meta, f, indent=2)
        
        print(f"Results saved to {results_file}")
        print()
    
    # Final save
    trainer.save_checkpoint()
    print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
