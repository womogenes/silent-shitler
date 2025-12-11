# Research logs: debugging and improving CFR for Secret Hitler

## Initial performance discrepancy

Our investigation began with a puzzling discrepancy. During training, the CFR agent for liberals reported a respectable 51% win rate against random fascist opponents after 500,000 iterations. This seemed like a reasonable improvement over the baseline 28.6% win rate that random liberal agents achieve. However, when we ran a comprehensive evaluation comparing all trained strategies against each other, the same CFR-Liberal agent only achieved a 28% win rate - essentially no better than random play. This 23-percentage-point gap was too large to be statistical noise. Something was fundamentally wrong.

The first hypothesis was that there might be a bug in how the evaluation script was using the CFR agent. We created a detailed debug script that would play out games and log every decision, showing whether the CFR agent was finding trained strategies or falling back to random play. The initial runs of this debug script confirmed the poor performance - liberals were winning only 28% of games. More concerningly, the script revealed that CFR was only finding trained infosets about 52% of the time, meaning nearly half of all decisions were falling back to uniform random strategies.

## Finding the hard-coding bug

The breakthrough came when we carefully compared the evaluation code in the original training script against our debug script. In train_cfr_liberal.py, buried at lines 249-250 of the evaluation function, we found a critical piece of logic that had been overlooked. The training evaluation was hard-coding liberal card selection - whenever a liberal player had the option to discard a fascist policy card, the code forced them to always choose that action rather than consulting the CFR strategy.

This same hard-coding appeared earlier in the training loop itself, at lines 172-178. During the CFR traversal, whenever the game phase was card selection and a liberal player could discard a fascist card, the code would immediately return without updating any regrets or strategies for that decision point. This meant that CFR never actually learned how to make card selection decisions - these critical game moments were completely bypassed during training.

When our original debug script tried to use pure CFR for all decisions, including card selection, it exposed this gap. The CFR agent had no learned strategy for these states, so it defaulted to uniform random selection. This explained the performance collapse from 51% to 28%. Once we added the same hard-coding to our debug script to match the training evaluation, the win rate jumped back up to the expected 51%.

## State space coverage analysis

With the immediate mystery solved, we dug deeper into why CFR was performing so poorly overall, achieving only a modest improvement from 28.6% to 31% against random opponents despite 500,000 training iterations. The debug transcripts revealed a troubling pattern. Early in games, CFR would consistently find trained infosets and use learned strategies. But as games progressed into the middle and late stages, the hit rate would plummet, with long sequences of decisions falling back to random play.

The root cause was the enormous state space created by the information set abstraction. The original implementation tracked five features per player: fascist policies played as president, fascist policies as chancellor, liberal policies as president, liberal policies as chancellor, and claim conflicts. Even with bucketing (capping each feature at 2+), this created 162 possible feature combinations per player. With five players, the theoretical space was 162^5, or about 11.6 billion possible information sets for liberals alone.

After 500,000 training iterations, the CFR agent had only encountered 366,256 unique infosets - a mere 0.003% coverage of the theoretical state space. This meant that in actual gameplay, especially as games developed unique histories, the agent would increasingly encounter states it had never seen before. Our debug logs showed games with coverage ranging from a dismal 29% to a best-case 85%, with an average around 52%. The agent was essentially playing half-random, which explained why it barely improved over the baseline.

## Code architecture problems

While investigating the CFR implementation, we uncovered significant architectural problems that violated basic software engineering principles. Unlike the clean PPO implementation where algorithm code lived in the agents directory and training scripts merely orchestrated, the CFR codebase was a mess of redundancy and poor separation of concerns.

We found three different CFR agent implementations scattered across the codebase. There was a base CFRAgent class in agents/cfr/cfr_agent.py, a CFRGameAgent wrapper in experiments/cfr/train_cfr.py, and a completely separate CFRPlusAgent implementation embedded in the 480-line experiments/cfr/train_cfr_liberal.py file. Even more puzzling, there was a well-structured CFRTrainer class in agents/cfr/trainer.py that was never imported or used by any training script.

The training scripts themselves were monolithic, mixing algorithm implementation, training loops, evaluation, logging, and checkpointing all in single files. The train_cfr_liberal.py script alone was 480 lines of intertwined code. This violated the principle of separation of concerns and made the code difficult to understand, debug, and modify. It also led to code duplication, as each training variant reimplemented similar functionality rather than composing reusable components.

## Learning from poker AI

To understand how to handle such large state spaces, we researched how poker AI systems like Libratus and Pluribus achieved superhuman performance despite Texas Hold'em having 10^160 states - vastly larger than our 10^10 state Secret Hitler game. The key insight was that successful game AI doesn't try to visit every possible state. Instead, it uses multiple techniques to handle the complexity.

Poker AI employs information abstraction to group similar hands into buckets, reducing billions of card combinations to thousands of strategic categories. It uses action abstraction to limit betting to key amounts rather than considering every possible bet size. Most importantly, modern poker AI uses neural networks to generalize between similar states, allowing the system to play well in situations it has never explicitly encountered during training.

The breakthrough systems also use sophisticated Monte Carlo CFR variants that sample the game tree rather than traversing it exhaustively. They employ depth-limited solving, maintaining a coarse blueprint strategy for early game positions while doing real-time fine-grained solving for the current decision point. This hierarchical approach allows them to balance computational resources between broad strategic planning and tactical precision.

## Implementing a simplified abstraction

Based on these insights, we implemented a dramatically simplified state abstraction to improve CFR's coverage. Instead of tracking five features per player with complex bucketing schemes, we reduced each player to a single suspicion level: clean (never played fascist), suspicious (played 1-2 fascist policies), or dirty (played 3+ fascist policies). This reduced the per-player combinations from 162 to just 3, shrinking the total state space from 11.6 billion to 3.6 million states - a 3,253-fold reduction.

We created a new infoset_simple.py module implementing this coarser abstraction, along with a modified training script train_cfr_liberal_simple.py that uses it. The new implementation also provides better instrumentation, continuously reporting the state space coverage percentage during training. With the smaller state space, we expect CFR to achieve 30-60% coverage after 500,000 iterations, compared to the previous 0.003%. This should translate to an 80-95% infoset hit rate during gameplay, dramatically reducing the amount of random fallback behavior.

## Game dynamics and balance

Through this deep dive into CFR's struggles, we gained valuable insights about Secret Hitler's fundamental dynamics. The game is heavily biased toward fascists, who win 71.4% of games between random players. This bias stems from multiple advantages: fascists have perfect information (knowing all roles), a deck advantage (11 of 17 policy cards are fascist), and multiple win conditions (6 fascist policies or Hitler being elected chancellor after 3 fascist policies).

Liberals face an uphill battle with only 6 of 17 policy cards in their favor, no information about who their teammates are, and the constant threat of accidentally electing Hitler. The fact that CFR could only improve liberal performance from 28.6% to 31% against random opponents, despite the fascist CFR achieving 82-84% win rates, highlights this fundamental asymmetry.

The hard-coding of liberal card play, which initially seemed like a bug, turned out to be strategically correct. Since fascists know all roles anyway, there's no deception value in liberals ever playing fascist policies. The hard-coding appropriately simplifies the strategy space while maintaining optimal play. This is an important lesson: sometimes domain knowledge should override learning, especially when certain actions are provably dominated.

## Technical appendix

### State space calculations

Original abstraction:
- Per player: 3 × 3 × 3 × 3 × 2 = 162 combinations (fasc_as_prez × fasc_as_chanc × lib_as_prez × lib_as_chanc × conflicts)
- Five players: 162^5 ≈ 11.6 billion combinations
- Additional factors: roles, policies, phase, etc.
- Total liberal state space: ~11.6 billion

Simplified abstraction:
- Per player: 3 suspicion levels
- Five players: 3^5 = 243 combinations
- With other factors: 3,061,800 liberal states + 504,000 fascist states = 3,565,800 total
- Reduction factor: 3,253x

### Coverage analysis

Original CFR after 500k iterations:
- Infosets visited: 366,256
- State space: 11.6 billion
- Coverage: 0.003%
- Gameplay hit rate: 51.8% average, ranging from 29.2% to 84.8%

Expected with simplified abstraction:
- Expected infosets: 1-2 million
- State space: 3.6 million
- Expected coverage: 30-60%
- Expected hit rate: 80-95%

### Performance comparison

All evaluations against random opponents, 500 games each:
- Random liberals: 28.6% win rate (baseline)
- CFR-Liberal (original): 31.0% win rate
- CFR-Standard (self-play): 28.4% win rate as liberals
- CFR-Standard as fascists: 82.6% win rate
- CFR-Liberal as fascists (playing randomly but counted): 78.8% win rate

The data clearly shows that CFR learning is much more effective for the fascist team, likely due to their information advantage and simpler coordination requirements.