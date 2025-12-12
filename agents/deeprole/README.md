# DeepRole Training Instructions

## Overview

DeepRole is a multi-agent reinforcement learning algorithm that combines Counterfactual Regret Minimization with deep neural networks. The implementation follows the approach from Serrino et al., adapted for 5-player Secret Hitler. The core innovation is using neural networks to evaluate game positions while maintaining belief distributions over hidden role assignments.

## Architecture

The DeepRole system consists of several interconnected components. The role assignment manager handles the enumeration of all 20 possible role assignments for 5 players (3 liberals, 1 fascist, 1 Hitler). The belief tracker maintains and updates probability distributions over these assignments using Bayesian updates and deductive reasoning specific to Secret Hitler. The vector-form CFR implementation operates on belief vectors rather than specific role assignments, allowing efficient computation across all possible worlds simultaneously.

The neural network architecture uses a win probability layer that learns to predict liberal win probability for each role assignment, then transforms these into player-specific values based on their roles. This interpretable intermediate representation helps the network learn more effectively than directly predicting values.

## Data Generation Process

The training data generation follows Algorithm 4 from the paper, which samples diverse game situations rather than simulating complete games. This approach is crucial for computational feasibility and training efficiency.

### Sophisticated Terminal Value Computation

Terminal states (5L or 6F) require special handling to generate diverse, strategic training data. The implementation uses a sophisticated CFR-based approach rather than simple belief-weighted averaging. For each terminal state sample, the system samples a plausible game history leading to that terminal state, including which governments succeeded, how players voted, and what policies were enacted. It then creates a game state one or two rounds before the terminal position and runs CFR from that position to compute strategic values that account for counterfactual reasoning and optimal play.

This approach produces highly diverse terminal values because different game histories and belief states lead to different strategic considerations. The values reflect what optimal play would look like given the specific context, not just static payoffs based on role assignments. While computationally more expensive than simple averaging, this sophistication is essential for preventing overfitting and ensuring the neural networks learn meaningful patterns.

To generate training data for the complete game, use the provided script:

```bash
# Full training (40-60 hours on 32-core machine)
uv run python agents/deeprole/generate_data.py \
    --samples 10000 \
    --cfr-iterations 1500 \
    --cfr-delay 500 \
    --workers 32 \
    --output trained_networks.pkl

# Quick test mode (30-60 minutes)
uv run python agents/deeprole/generate_data.py --quick

# Verify data diversity before full training
uv run python agents/deeprole/verify_terminal_diversity.py
```

Or programmatically:

```python
from agents.deeprole.backwards_training import BackwardsTrainer
import torch
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

trainer = BackwardsTrainer(num_workers=32)  # Adjust based on CPU cores

# Generate data for all game states using backwards training order
# This will take significant time - recommend running on cluster
trainer.train_all_networks(
    samples_per_stage=10000,  # Number of samples per (lib, fasc) state
    cfr_iterations=1500,      # CFR iterations per sample (paper default)
    cfr_delay=500,            # Averaging delay (paper default)
    save_path="trained_networks.pkl"
)
```

The backwards training procedure starts from terminal states where the outcome is known and works backwards through the game tree. For terminal states like (5L, 0F) where liberals have won, the values are computed directly based on role assignments without needing CFR. For non-terminal states, the system samples random game situations using Dirichlet distributions to create diverse belief states, then runs CFR with the already-trained networks from later game stages as leaf evaluators.

The data generation is computationally intensive. With the paper's recommended settings of 10,000 samples per stage and 1,500 CFR iterations per sample, generating data for all 42 game states will require substantial computation time. The multiprocessing implementation distributes this work across available CPU cores, but expect several hours even on a powerful machine.

## Training Process

Once data is generated, the neural networks are trained using standard supervised learning. The training process for each game state involves fitting a network to predict the CFR-computed values given the current president and belief distribution.

For GPU training, you can modify the training loop in `backwards_training.py` to use CUDA:

```python
# In train_network method, after creating the network
network = ValueNetwork()
if torch.cuda.is_available():
    network = network.cuda()
    print(f"Training on GPU: {torch.cuda.get_device_name()}")

# Convert inputs to CUDA in the training loop
inputs = inputs.cuda()
targets = targets.cuda()
```

The recommended training hyperparameters from the paper are 3000 epochs per network with Adam optimizer at learning rate 0.001. The current implementation uses 100 epochs for faster iteration during development. For production training, modify line 215 in `backwards_training.py`:

```python
n_epochs = 3000  # Paper's recommendation
```

## Parallelization for Cluster

The data generation is embarrassingly parallel across different game states. To run on a cluster, you can split the work by game state:

```python
def train_single_game_state(lib_policies, fasc_policies):
    """Train network for single game state - suitable for cluster job."""
    trainer = BackwardsTrainer(num_workers=8)  # Per-node parallelism

    # Load any previously trained networks for later stages
    if Path("partial_networks.pkl").exists():
        trainer.networks.load("partial_networks.pkl")

    # Generate data for this specific state
    training_data = trainer.generate_training_data(
        lib_policies, fasc_policies,
        n_samples=10000,
        cfr_iterations=1500,
        cfr_delay=500
    )

    # Train network
    network = trainer.train_network(training_data, lib_policies, fasc_policies)

    # Save this network
    save_path = f"network_{lib_policies}L_{fasc_policies}F.pth"
    torch.save(network.state_dict(), save_path)

    return save_path

# Submit separate jobs for each game state
# Start with terminal states, then work backwards
terminal_states = [(5, i) for i in range(6)] + [(i, 6) for i in range(5)]
```

## Memory Requirements

Each training sample contains a 25-dimensional input (5 for one-hot president, 20 for belief) and 5-dimensional output (values for each player). With 10,000 samples per game state and 42 game states, the complete dataset requires approximately 2GB of memory. The CFR solving process during data generation has higher transient memory usage, potentially requiring 8-16GB RAM per worker process.

## Verification

After training, verify the networks by testing on known positions:

```python
from agents.deeprole.networks import NetworkEnsemble

ensemble = NetworkEnsemble()
ensemble.load("trained_networks.pkl")

# Test on terminal state - should give deterministic values
lib_win_network = ensemble.networks[(5, 0)]
president = torch.tensor(0)
belief = torch.ones(20) / 20  # Uniform belief

with torch.no_grad():
    values = lib_win_network(president, belief)
    print(f"Liberal win values: {values[0]}")  # Should be positive for liberal

# Test on mid-game state
mid_game_network = ensemble.networks[(2, 2)]
values = mid_game_network(president, belief)
print(f"Mid-game values: {values[0]}")  # Should be closer to 0
```

## Expected Training Time

On a modern CPU with 32 cores, expect the following approximate times:

Data generation for all 42 game states with 10,000 samples each will take 40-60 hours with the sophisticated terminal value computation. Terminal states generate at approximately 10-20 samples per second due to CFR evaluation, while non-terminal states process faster at 5-10 samples per second depending on game complexity. The improved data quality from CFR-based terminal values justifies the additional computation time by preventing overfitting and producing more strategic training data.

Neural network training with 3,000 epochs per network takes approximately 10 minutes per game state on GPU, or about 7 hours total for all networks. The time is dominated by data generation rather than network training.

For faster iteration during development, use the --quick flag which reduces samples to 20 per stage and CFR iterations to 50. This completes in 30-60 minutes and is sufficient for testing the pipeline. For production training, use the full parameters to achieve the paper's performance level.

## Using Trained Networks

Once training completes, the networks can be used for gameplay:

```python
from agents.deeprole.gameplay_solver import DeepRoleSolver

solver = DeepRoleSolver(network_path="trained_networks.pkl")

# During game, get action for player
action = solver.get_action(
    env,
    player_idx=current_player,
    belief=current_belief,
    cfr_iterations=50,  # Real-time solving
    cfr_depth=3,        # Depth limit
    temperature=0.1     # Near-greedy
)
```

The solver runs short CFR searches using the neural networks to evaluate positions beyond the search depth, enabling real-time decision making while maintaining the theoretical guarantees of CFR.

## Troubleshooting

If you encounter memory errors during data generation, reduce the number of worker processes or samples per batch. The CFR solving can be memory intensive for complex game states.

If neural network training loss doesn't decrease, verify that the data generation is producing meaningful values by inspecting a few samples manually. Terminal states should have clear positive/negative values for liberals/fascists.

For numerical stability issues, the implementation already includes epsilon terms in probability calculations and proper normalization of belief distributions. If problems persist, check that belief vectors sum to 1.0 and reach probabilities remain non-negative.

## Citation

This implementation is based on "DeepRole: Multi-Agent Counterfactual Regret Minimization in Imperfect Information Games" by Serrino et al. The key algorithmic contributions are the vector-form CFR with belief tracking, the win probability neural network layer, and the backwards training procedure that enables learning without full game tree expansion.