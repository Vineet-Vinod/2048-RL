# 2048: expectimax and reinforcement learning

This is a from-scratch experiment in teaching an agent to play 2048. It began
as a Bellman-style search, moved to model-free deep Q-learning, and ended with a
hybrid that uses exact expectimax search with a neural value function at the
leaves.

The main lesson was fairly blunt: expectimax performed best. The hybrid came
second. Pure reinforcement learning performed badly for the amount of compute
used.

## What worked

| Approach | Result in this project |
| --- | --- |
| Heuristic expectimax | Best. It made the strongest moves and reached 2048. |
| Expectimax with a learned value function | Second. It learned much faster than the pure RL agent and could reach 1024 early in training. |
| Model-free DQN | Worst. Training took a long time, scores were unstable, and the agent did not collect enough useful experience for the size of the state space. |

Pure DQN was not sample efficient here. A game provides one narrow trajectory
through an enormous set of possible boards. The agent had to learn both the
consequences of each action and which resulting boards were useful from those
sampled trajectories. Even tens or hundreds of thousands of games did not
provide good coverage, and optimizer time made collecting and learning from
more rollouts expensive.

2048 gives us its transition model almost for free. After a move, a new tile is
either a 2 with probability 0.9 or a 4 with probability 0.1, placed uniformly
in an empty cell. Expectimax can enumerate those outcomes instead of waiting to
observe them through repeated games. In this setting, using the known model was
far more effective than asking a model-free agent to rediscover it.

The hybrid keeps that exact search and asks a small neural network to estimate
the value of boards at the search-depth limit. Search handles the local action
and spawn possibilities. The network tries to generalize beyond the states
that search can reach cheaply.
