"""CFR training loop for Silent Shitler."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shitler_env"))

from game import ShitlerEnv
from .cfr_agent import CFRAgent, outcome_sampling_cfr


class CFRTrainer:
    """
    Trainer for External Sampling MCCFR.
    
    Runs self-play iterations, alternating traverser.
    """
    
    def __init__(self, checkpoint_dir=None):
        self.cfr_agent = CFRAgent()
        self.iterations = 0
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, num_iterations, checkpoint_every=1000, verbose=True):
        """Run training iterations."""
        for i in range(num_iterations):
            self.iterations += 1
            
            # Create fresh game
            env = ShitlerEnv()
            env.reset()
            
            # Alternate traverser across all 5 players
            traverser_idx = self.iterations % 5
            
            # Run MCCFR traversal
            outcome_sampling_cfr(env, self.cfr_agent, traverser_idx)
            
            if verbose and self.iterations % 100 == 0:
                num_infosets = len(self.cfr_agent.regret_sums)
                print(f"Iteration {self.iterations}: {num_infosets} infosets")
            
            if checkpoint_every and self.iterations % checkpoint_every == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self, path=None):
        """Save agent state to disk."""
        if path is None:
            if self.checkpoint_dir is None:
                return
            path = self.checkpoint_dir / f"cfr_iter_{self.iterations}.pkl"
        
        data = {
            "iterations": self.iterations,
            "regret_sums": dict(self.cfr_agent.regret_sums),
            "strategy_sums": dict(self.cfr_agent.strategy_sums),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load agent state from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.iterations = data["iterations"]
        self.cfr_agent.regret_sums = defaultdict(
            lambda: defaultdict(float), 
            {k: defaultdict(float, v) for k, v in data["regret_sums"].items()}
        )
        self.cfr_agent.strategy_sums = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data["strategy_sums"].items()}
        )
        print(f"Loaded checkpoint from {path} (iteration {self.iterations})")
    
    def get_exploitability_estimate(self, num_games=100):
        """
        Rough exploitability estimate via random opponent.
        
        TODO: implement proper best response calculation
        """
        wins = {"lib": 0, "fasc": 0}
        for _ in range(num_games):
            env = ShitlerEnv()
            env.reset()
            
            while not all(env.terminations.values()):
                agent = env.agent_selection
                obs = env.observe(agent)
                action_space = env.action_space(agent)
                
                # Use CFR strategy
                action = self.cfr_agent.get_action(obs, action_space, env.phase)
                env.step(action)
            
            # Count wins
            for agent in env.agents:
                if env.rewards[agent] > 0:
                    if env.roles[agent] == "lib":
                        wins["lib"] += 1
                    else:
                        wins["fasc"] += 1
                    break
        
        return wins


# Convenience function
def defaultdict(factory, initial=None):
    """Helper to recreate defaultdicts from loaded data."""
    from collections import defaultdict as dd
    d = dd(factory)
    if initial:
        d.update(initial)
    return d


if __name__ == "__main__":
    trainer = CFRTrainer(checkpoint_dir="checkpoints/cfr")
    trainer.train(num_iterations=10000, checkpoint_every=1000)
    
    print("\nEvaluating...")
    wins = trainer.get_exploitability_estimate(num_games=100)
    print(f"Win rates over 100 games: {wins}")
