# DeepRole Agent Usage

## Quick Start

After training networks with `generate_data.py`, you can test the DeepRole agent:

```bash
# Quick test (10 games)
python run_deeprole_eval.py

# Full evaluation (100 games)
python test_deeprole.py --games 100

# Fast evaluation with simple agent
python test_deeprole.py --games 100 --simple

# Debug single game
python test_deeprole.py --single
```

## Agent Classes

### `DeepRoleAgent`
Full implementation with CFR-based decision making:
- Uses trained neural networks for value estimation
- Runs CFR iterations for each decision
- Maintains belief state over hidden roles
- **Slower but more accurate**

### `SimpleDeepRoleAgent`
Simplified version for faster play:
- Uses trained networks for guidance
- Simple heuristics for decisions
- No CFR computation
- **Much faster but less sophisticated**

## Integration with Your Code

```python
from agents.deeprole.deeprole_agent import DeepRoleAgent

# Create agent with trained networks
agent = DeepRoleAgent(
    networks_path="trained_networks.pkl",
    cfr_iterations=50,  # Fewer for speed, more for accuracy
    max_depth=3         # Search depth
)

# Use in game
obs = env.observe(agent_name)
action = agent.get_action(obs)
env.step(action)
```

## Using with eval_agent.py

The agents are compatible with the standard evaluation interface:

```python
from shitler_env.eval_agent import evaluate_agents
from agents.deeprole.deeprole_agent import SimpleDeepRoleAgent

# Create agent classes
agent_classes = [
    lambda: SimpleDeepRoleAgent("trained_networks.pkl"),
    SimpleRandomAgent,
    SimpleRandomAgent,
    SimpleRandomAgent,
    SimpleRandomAgent,
]

# Evaluate
results = evaluate_agents(agent_classes, num_games=100)
```

## Performance Considerations

### Speed vs Quality Trade-offs

| Setting | CFR Iterations | Max Depth | Speed | Quality |
|---------|---------------|-----------|--------|---------|
| Fast | 10 | 2 | ~0.1s/decision | Low |
| Balanced | 50 | 3 | ~0.5s/decision | Medium |
| Quality | 200 | 4 | ~5s/decision | High |
| Paper | 1500 | 10 | ~60s/decision | Highest |

### Recommended Settings

- **Testing/Development**: Use `SimpleDeepRoleAgent`
- **Evaluation**: `DeepRoleAgent` with 50 iterations, depth 3
- **Competition**: `DeepRoleAgent` with 200+ iterations, depth 4+

## Expected Performance

With properly trained networks:
- **vs Random agents**: Should win 60-70% as liberals
- **vs Simple heuristics**: Should win 55-65% as liberals
- **Self-play**: Should converge to ~50% win rate

Without trained networks (fallback):
- Performs similar to random agent

## Troubleshooting

### No networks found
- Check that `trained_networks.pkl` exists
- Or specify path: `DeepRoleAgent("path/to/networks.pkl")`

### Agent plays poorly
- Check network training completed
- Increase CFR iterations
- Verify belief tracking is working

### Agent is too slow
- Reduce CFR iterations
- Reduce max_depth
- Use SimpleDeepRoleAgent instead

## Files

- `deeprole_agent.py`: Agent implementations
- `test_deeprole.py`: Comprehensive testing script
- `run_deeprole_eval.py`: Quick evaluation script
- `trained_networks.pkl`: Trained neural networks (after running generate_data)

## Next Steps

1. Train networks: `python agents/deeprole/generate_data_3hr.py`
2. Test agent: `python run_deeprole_eval.py`
3. Tune parameters for your use case
4. Integrate into your tournament/evaluation framework