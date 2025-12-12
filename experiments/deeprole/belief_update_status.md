# DeepRole Belief Update Implementation Status

## What the Paper Requires

According to the DeepRole paper (Serrino et al.), belief updates should follow equation (1):

```
b(ρ|h) ∝ b(ρ)(1 - 1{h ⊢ ¬ρ}) ∏_{i∈1...p} π^σ_i(I_i(h, ρ))
```

This requires:
1. **Prior belief** b(ρ)
2. **Logical consistency check** (1 - 1{h ⊢ ¬ρ})
3. **Strategy-based update** ∏_i π_i(I_i(h, ρ))

## What We've Implemented

### ✅ Logical Deductions (Implemented)
- Zeros out role assignments that are logically inconsistent with observations
- Examples:
  - If we know our own role, eliminate inconsistent assignments
  - If someone was chancellor with 3+ fascist policies and game didn't end, they're not Hitler
  - If someone was executed and game ended, they were Hitler
  - If someone was executed and game continued, they weren't Hitler

### ❌ Strategy-Based Updates (NOT Implemented)
- Requires the strategies π computed during CFR
- Would allow Bayesian inference from:
  - Voting patterns
  - Policy outcomes
  - Player claims
  - Mission participation

## The Fundamental Challenge

**During training/self-play**: We have all player strategies from CFR, so we can do full belief updates.

**During actual play**: We don't know opponent strategies, so we CAN'T do the full update.

The paper uses strategies computed during CFR as the likelihood in Bayes' rule, but this assumes we know how opponents play. When facing unknown opponents, we lack this critical information.

## Current Implementation

```python
# In belief_update.py
def update_belief(self, belief, obs, strategies=None):
    # 1. Apply logical deductions ✅
    new_belief = self._apply_logical_deductions(new_belief, obs)

    # 2. Strategy-based update ❌
    if strategies is not None:
        # Would implement: b[ρ] * ∏_i π_i(I_i(h, ρ))
        # But we don't have opponent strategies during play
        pass
```

## Impact on Performance

Without strategy-based updates:
- Belief remains nearly uniform except for hard logical constraints
- Can't learn from subtle behavioral patterns
- Can't update based on voting coalitions
- Can't infer from policy choices

This explains why DeepRole underperforms in our tests - it's missing a critical component that the paper assumes is available.

## Possible Solutions

1. **Store self-play strategies** - Use strategies from training as proxy (biased, may not match actual opponents)
2. **Assume uniform random** - Treat opponents as playing randomly (weak assumption)
3. **Online learning** - Learn opponent models during play (complex, requires many games)
4. **Population-based** - Train against diverse agent population (more robust but still limited)

## Honesty About the Gap

The paper doesn't fully address how to handle belief updates when playing against unknown opponents. The experiments in the paper involve:
- Self-play (where strategies are known)
- Playing against specific baselines (CFR, LogicBot, etc.)
- Human play (but doesn't explain how beliefs are updated without knowing human strategies)

This is a fundamental limitation of applying CFR-based methods to real-world play against unknown opponents.

## Code Location

- **Implementation**: `/agents/deeprole/belief_update.py`
- **Integration**: `/agents/deeprole/deeprole_agent.py` (lines 128-149)
- **Tests**: `/experiments/deeprole/test_belief_update.py`

## Summary

We've implemented the **logical deduction** part of belief updates correctly, but the **strategy-based updates** require information (opponent strategies) that we don't have during actual play. This is a fundamental challenge that the paper doesn't fully resolve.