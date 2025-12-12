A DeepRole depth-limited CFR
Algorithm 1 DeepRole depth-limited CFR
1: INPUT h (root public game history); b (root public belief); n (# iterations); d (averaging delay);
NN[h] (neural networks that approximate CFVs from h)
Init regrets ∀I, rI [a] ← 0, Init cumulative strategies ∀I, sI [a] ← 0
2: procedure SOLVESITUATION(h, b, n, d)
3: ~u1...p ← ~0
4: for i = 1 to n do
5: wi ← max(i − d, 0)
6: ~u1...p ← ~u1...p+MODIFIEDCFR+(h, b, wi,~11...p)
7: end for
8: return ~u1...p / ∑ wi
9: end procedure
10: procedure MODIFIEDCFR+(h, b, w, ~π1...p)
11: if h ∈ Z then
12: return TERMINALCFVS(h, b, ~π1...p)
13: end if
14: if h ∈ NN then
15: return NEURALCFVS(h, b, ~π1...p)
16: end if
17: ~u1...p ← ~0
18: for i ∈ P ′(h) do . A strategy must be calculated for all moving players at public history h
19: ~Ii ← lookupInfosetsi(h)
20: ~σi ← regretMatching+(~Ii)
21: end for
22: for public observation o ∈ O(h) do
23: ~a1...p ← deduceActions(h, o)
24: for i ∈ P ′(h) do
25: ~πi ← ~σi[~ai] ~πi
26: end for
27: ~u′
1...p ← MODIFIEDCFR+(ho, b, w, ~π1...p)
28: for each player i do
29: if i ∈ P ′(h) then
30: ~mi[~ai] ← ~mi[~ai] + ~ui
31: ~ui ← ~ui + ~σi[~ai] ~u′
i
32: else
33: ~ui ← ~ui + ~u′
i
34: end if
35: end for
36: end for
37: for i ∈ P ′(h) do . Similar to line 18, we must perform these updates for all moving players
38: for I ∈ ~Ii do
39: for a ∈ A(I) do
40: rI [a] ← max(rI [a] + ~mi[a][I] − ~ui[I], 0)
41: sI [a] ← sI [a] + ~πi[I]~σi[I][a]w
42: end for
43: end for
44: end for
45: return ~u1...p
46: end procedure
12Algorithm 2 Terminal value calculation
1: procedure TERMINALCFVS(h, b, ~π1...p)
2: ~v1...p[·] ← 0 . Initialize factual values
3: bterm ← CALCTERMINALBELIEF(h, b, ~π1...p)
4: for i = 1 to p do
5: for ρ ∈ bterm do
6: ~vi[Ii(h, ρ)] ← ~vi[Ii(h, ρ)] + bterm[ρ]ui(h, ρ)
7: end for
8: end for
9: return ~v1...p/~π1...p . Convert factual to counterfactual
10: end procedure
11: procedure NEURALCFVS(h, b, ~π1...p)
12: bterm ← CALCTERMINALBELIEF(h, b, ~π1...p)
13: w ← ∑
ρ bterm[ρ]
14: ~v1, ~v2, . . . , ~vp ← NN[h](h, bterm/w) . Call NN with normalized belief
15: return w~v1...p/~π1...p . Convert factual to counterfactual
16: end procedure
17: procedure CALCTERMINALBELIEF(h, b, ~π1...p)
18: for ρ ∈ b do
19: bterm[ρ] ← b[ρ] ∏
i ~πi(Ii(h, ρ))
20: bterm[ρ] ← bterm[ρ](1 − 1{h ` ¬ρ}) . Zero beliefs that are logically inconsistent
21: end for
22: return bterm
23: end procedure
B Value network training
We generate training data for the deep value networks by using CFR to solve each part of the game from a
random sample of starting beliefs. By working backwards from the end of the game, trained networks from later
stages enable data generation using CFR at progressively earlier stages. This progressive back-chaining follows
the dependency graph of proposals shown on the left side of Figure 1. This generalizes the procedure used to
generate DeepStack’s value networks [27].
For each network, we sampled 120, 000 game situations (θ ∈ Θ) to be used for training and validation. For
each sample, CFR ran for 1500 iterations, skipping the first 500 during averaging. The neural networks were
each trained for 3000 epochs (batch size of 4096) using the Adam optimizer with a mean squared error loss on
~V . Training hyperparameters and weight initializations used Keras defaults. 10% of the data was set aside for
validation. Training on 480 CPU cores, 480 GB of memory, and 1 GPU took roughly 48 hours to produce the
networks for every stage in the game.
C Comparison Agents
CFR CFR denotes an agent using a strategy trained by external sampling MCCFR with a hand-built imperfect-
recall abstraction, used to reduce the size of Avalon’s immense game tree. We bucket information sets for players
based on their initial information set (their role and who they see) and a set of hand-chosen game metrics: the
round number, the number of failed missions each player has participated in, and the number of times a player
has proposed a failing mission. We trained the bot until we observed decayed performance of the bot in self-play.
In total, CFRBot was trained for 6,000,000 iterations.
LogicBot LogicBot is an agent that plays a hand-crafted pre-set strategy derived from our intuition of playing
Avalon with real people. During play, LogicBot keeps a list of possible role assignments that are logically
consistent with the observations it has made. As resistance, it randomly samples an assignment and proposes a
mission using the resistance players in that assignment. It votes up proposals if and only if the proposed players
and the proposer are resistance in a randomly sampled assignment or if it is the final proposal in the round. As
spy, it proposes randomly, votes opposite to resistance players, and selects merlin randomly.
Random Our random agent selects an action uniformly from the available actions.
13Algorithm 3 Backwards training
1: INPUT P1...n: Dependency-ordered list of game parts.
2: INPUT Θ1...n: For each game part, a distribution over game situations.
3: INPUT d: The number of training datapoints generated per game partition.
4: OUTPUT N1...n: n trained neural value networks, one for each game part.
5: procedure ENDTOENDTRAIN(P1...n, Θ1...n, d) . Train a neural network for each game
partition
6: for i = 1 to n do
7: x, y ← GENERATEDATAPOINTS(Pi, Θi, N1...i−1)
8: Ni ← TRAINNN(x, y)
9: end for
10: return N1...n
11: end procedure
12: procedure GENERATEDATAPOINTS(d, S, Θ, N1...k) . Given a game partition, it’s distribution
over game situations, and the NNs needed to limit solution depth, generate d datapoints.
13: for i = 1 to d do
14: θi ∼ Θ . Sample a game situation from the distribution
15: vi ← SOLVESITUATION(S, θi, N1...k) . Solve that game situation for every player’s
values, using previously trained neural networks to solution depth.
16: end for
17: return θ1...d, v1...d . Return all training datapoints
18: end procedure
Algorithm 4 Game Situation Sampler
1: INPUT s: The number of succeeds.
2: INPUT f : The number of fails.
3: OUTPUT p, b: A random game situation from this game part, consisting of a proposer and a
belief over the roles.
4: procedure SAMPLESITUATION(s, f )
5: I ← SAMPLEFAILEDMISSIONS(s, f ) . Uniformly sample f failed missions
6: E ← EVILPLAYERS(I) . Calculate evil teams consistent with the missions
7: P (E) ∼ Dir(~1|E|) . Sample probability of each evil team
8: P (M ) ∼ Dir(~1n) . Sample probability of being Merlin for all players
9: b ← P (E) ⊗ P (M ) . Create a belief distribution using P (E) and P (M )
10: p ∼ unif{1, n} . Sample a proposer uniformly over all the players
11: return p, b
12: end procedure
14ISMCTS & MOISMCTS We also evaluate our bot against opponents playing using the ISMCTS family
of algorithms. Specifically, we evaluate our bot against the single-observer ISMCTS (ISMCTS) algorithm shown
in [ 11, 12, 43], as well as the improved multiple-observer version of the algorithm (MOISMCTS). Each variant
used 10,000 iterations per move.
D State space calculation
Unlike large two-player games like Poker, Go, or Chess, Avalon’s complexity lies in the combinatorial explosion
that comes with having 5 players, four role types (Spy, Resistance, Merlin, Assassin), and a large number
observable moves. We lower bound the number of information sets by just considering the longest possible
game. The longest possible game lasts five rounds with each round requiring five proposals. Each proposal can
made in 10 different ways by choosing which 2 or 3 players out of 5 should go on the mission. From there, there
are 16 ways proposals 1-4 can be voted down and 16 ways proposal 5 can be voted up. Thus, a lower bound
on the number of information sets is (10 ∗ 16)5∗5 ≈ 1056 which does not consider shorter games or any of the
private information.
