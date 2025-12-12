# DeepRole: How It Actually Works at Play-Time

## The Complete Flow

DeepRole doesn't use pre-computed strategies from training. Instead:

1. **At each decision point during play:**
   ```
   Current game state → Run CFR (1500 iterations) → Compute strategies for ALL players
                                     ↓
                            Use neural networks for leaf evaluation
   ```

2. **CFR computes strategies for everyone:**
   - Not just for us, but for all players
   - Based on the current belief distribution
   - These are "equilibrium" strategies given current information

3. **Use CFR strategies to update beliefs:**
   ```
   b(ρ|h) ∝ b(ρ) * ∏_i π_i(I_i(h, ρ))
   ```
   Where π_i comes from the real-time CFR we just ran!

4. **Select action according to our CFR strategy**

## What We're Missing

The current implementation:
- ✅ Runs CFR at each decision
- ✅ Computes strategies (stored in `strategy_sums`)
- ❌ Doesn't extract strategies for belief updates
- ❌ Doesn't use strategies to update beliefs

## The Fix Needed

```python
# After running CFR
values = self.cfr.solve_situation(env, belief, ...)

# Extract strategies from CFR
strategies = self.cfr.get_average_strategies()  # Need to implement this

# Update belief using these strategies
new_belief = update_belief_with_strategies(belief, obs, strategies)

# Use updated belief for next decision
```

## Why This Makes Sense

- CFR assumes all players are rational
- It computes what rational players would do
- We use these rational strategies as our "model" of opponents
- This allows Bayesian belief updates even against unknown opponents

## The Key Insight

DeepRole doesn't need to know actual opponent strategies. It uses CFR to compute what rational players WOULD do given the current information, and uses those computed strategies for belief updates.

This is brilliant because:
- Works against any opponent (not just those seen in training)
- Adapts to current game state
- Provides consistent belief updates
- No heuristics needed

## Current Bug

We're running CFR but NOT extracting and using the computed strategies for belief updates. That's why beliefs stay nearly uniform!