# Project report

## Requirements

- What project did you work on? What was the precise problem definition/project goal?
- How did you split the work among the team members?
- What approach did you take? How does it relate to existing approaches? What was known in the literature?
- What were the technical challenges? How did you try to resolve them?
- How did you evaluate your approach? For the team poker challenges, it is acceptable to refer to the leaderboard.
- What were the findings? How does the performance compare to other approaches (if applicable)? Are the results surprising or expected?
- What were the bottlenecks of the approach you selected? What future work could be done to overcome them?
- What are the main takeaways of your project?
- How does the project relate to what you learned in this class?

Other requirements:

- Max 10 pages long
- Include references

## Pre-proposal

\textbf{Project title.} Learning Algorithms for Silent Social Deduction Games

\textbf{Team members.} [REDACTED]

\textbf{Objectives.} Secret Hitler is a popular social deduction game with up to ten players. Players are randomly divided into a liberal and a fascist team, where the fascists know every player's role but the liberals do not. In short, the goal of the liberal team is to pass a number of \textit{liberal policies} while the goal of the fascist team is to pass a number of \textit{fascist policies} or have Hitler---a predetermined member of their team---elected Chancellor.

In standard play, players are allowed to communicate publicly. We will forgo this aspect of the game as it introduces elements of collaboration and strategic communication, which are difficult to control for. For simplicity and in order to focus on the more game theoretic elements, we will use a subset of the game rules listed in the \textbf{Appendix.}

Interesting dynamics that arise from the hidden-information nature of the game, such as players claiming conflicting information or Hitler playing liberal moves to get themselves elected Chancellor. We hope to elicit these behaviors in the algorithms we will develop to play the game.

\textbf{Summary.}

The game is guaranteed to end after at most 30 rounds because of the election tracker. The game state can therefore be represented as a finite-dimensional vector, from which we can train an algorithm to play the game. As the game tree is too large, we hope to use heuristic-based approaches such as PPO using an actor (policy) and critic (value) network to develop an optimal policy.

\textbf{Project Break.}

\begin{itemize}
\item \textbf{November 11.} We aim to finish creating a representation of our game.
\item \textbf{November 25.} We aim to have a baseline PPO algorithm implemented on the game.
\item \textbf{December 2.} We aim to finish our final report and create a presentation.
\end{itemize}

\textbf{Appendix: Simplified game rules.}

\begin{enumerate}
\itemsep-0.5em
\item There are five players, in seats numbered 1, 2, 3, 4, 5 around a table.
\item At the start of the game, roles are randomly assigned. There are three liberals, one fascist, and one Hitler. The fascist and Hitler learn who each other are.
\item A \textbf{deck} with 11 fascist policy cards and 6 liberal policy cards is shuffled and laid face-down on the table. Cards of the same alignment are interchangeable. There is a \textbf{discard pile} that initially starts empty.
\item The game proceeds in a series of \textbf{governments}. For each government:
\begin{enumerate}
\itemsep-0.5em
\item One player is named the \textbf{President}. This role rotates one seat per government. The president nominates a \textbf{Chancellor}, and all players simultaneously vote yes/no for this government.
\item If a majority of players vote yes, the President secretly draws three cards from the top of the deck, discards one and passes the remaining two to the Chancellor, who then discards one card and plays the final onto the table. After the cards are played, the President and Chancellor each publicly \textbf{claim} the cards that they saw.
\item If a majority of players vote no, the government \textbf{fails} and the \textbf{election counter} is incremented. If this counter ever reaches 3, the country descends into \textbf{chaos} and the top card from the policy deck is played onto the table, after which the counter resets. The counter also resets after every successful election.
\end{enumerate}
\item If the policy deck ever goes below three cards, it is mixed with the discard pile and re-shuffled.
\item If five liberal policies are passed, the liberals wins. If six fascist policies are passed, the fascists win. Some special rules occur after each fascist policy is played:
\begin{enumerate}
\itemsep-0.5em
\item After three fascist policies have been passed, if Hitler is ever elected Chancellor, the \textbf{fascist team wins}.
\item After the fourth and fifth fascist policies are passed, the President gets to \textbf{execute} a player of their choosing, who is now out of the game. If Hitler is killed, the liberals win.
\end{enumerate}
\end{enumerate}

\end{document}
