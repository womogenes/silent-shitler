# Learning Algorithms for Silent Social Deduction Games

## Objectives

Secret Hitler is a popular social deduction game with up to ten players. Players are randomly divided into a liberal and a fascist team, where the fascists know every player's role but the liberals do not. In short, the goal of the liberal team is to pass a number of _liberal policies_ while the goal of the fascist team is to pass a number of _fascist policies_ or have Hitler—a predetermined member of their team—elected Chancellor.

In standard play, players are allowed to communicate publicly. We will forgo this aspect of the game as it introduces elements of collaboration and strategic communication, which are difficult to control for. For simplicity and in order to focus on the more game theoretic elements, we will use a subset of the game rules listed in the **Appendix.**

Interesting dynamics that arise from the hidden-information nature of the game, such as players claiming conflicting information or Hitler playing liberal moves to get themselves elected Chancellor. We hope to elicit these behaviors in the algorithms we will develop to play the game.

## Summary

The game is guaranteed to end after at most 30 rounds because of the election tracker. The game state can therefore be represented as a finite-dimensional vector, from which we can train an algorithm to play the game. As the game tree is too large, we hope to use heuristic-based approaches such as PPO using an actor (policy) and critic (value) network to develop an optimal policy.

## Appendix: Simplified Game Rules

1. There are five players, in seats numbered 1, 2, 3, 4, 5 around a table.
2. At the start of the game, roles are randomly assigned. There are three liberals, one fascist, and one Hitler. The fascist and Hitler learn who each other are.
3. A **deck** with 11 fascist policy cards and 6 liberal policy cards is shuffled and laid face-down on the table. Cards of the same alignment are interchangeable. There is a **discard pile** that initially starts empty.
4. The game proceeds in a series of **governments**. For each government:
   1. One player is named the **President**. This role rotates one seat per government. The president nominates a **Chancellor**, and all players simultaneously vote yes/no for this government.
   2. If a majority of players vote yes, the President secretly draws three cards from the top of the deck, discards one and passes the remaining two to the Chancellor, who then discards one card and plays the final onto the table. After the cards are played, the President and Chancellor each publicly **claim** the cards that they saw.
   3. If a majority of players vote no, the government **fails** and the **election counter** is incremented. If this counter ever reaches 3, the country descends into **chaos** and the top card from the policy deck is played onto the table, after which the counter resets. The counter also resets after every successful election.
5. If the policy deck ever goes below three cards, it is mixed with the discard pile and re-shuffled.
6. If five liberal policies are passed, the liberals win. If six fascist policies are passed, the fascists win. Some special rules occur after each fascist policy is played:
   1. After three fascist policies have been passed, if Hitler is ever elected Chancellor, the **fascist team wins**.
   2. After the fourth and fifth fascist policies are passed, the President gets to **execute** a player of their choosing, who is now out of the game. If Hitler is killed, the liberals win.
