# Deep-Reinforcement-Learning
Deep Reinforcement learning project with hands-on experience implementing advanced version of the classic Tic Tac Toe game
Project 4 – TIC-TAC-TOE with Reinforcement Learning 
Student Name: Yair Lahad
Submission Date: 30/06/2025

Part 1 – Monte Carlo

1.a Solution on Colab code.
1.b Learning Rate: low LR Is stable and converges. Higher LR converges fast for less epochs (20k) yet unstable with more epochs.

 

Discount Factor is currently not used in Monte Carlo implementation but in general its effect:
 
Epsilon effect on wins/loss: could not directly understand strategy usage. A proper method to learn about epsilon is analyzing final performance.

 


Episode number impact on the agent overall performance: helps until converges around 40k episodes
 




1.c From the analysis on section 1.b and from parameter analysis done in code I choose the following:
	1. High Learning Rate = 0.5
	2. High Epsilon = 0.7 as seen in 1.b as the optimized value.

 
 


As seen on the Win/Loss/Draw rates, applying those parameters increases the overall performance of the Agent each time.





Part 2 – Q Learning 

2.a Colab code. This is one baseline for evaluation. 

 


2.b Agent learning haven't improved after changing the learning rate to 0.1. It lost more than baseline stats. Meaning it is less stable with worse convergence.

 


2.c
For 0.9 discount factor: Agent learns better by valuing future rewards less extremely than baseline.
even though training stats show worse win rate and higher exploration rate, agent wins more on evaluation, fit for short games like Tic-Tac-Toe.

 
For 0.5 discount factor: Agent learns poorly by not valuing future rewards enough.

 

2.d The ε = 1 agent learned passively from random episodes without ever exploiting its Q-values, so it lacked consistent reinforcement of successful strategies. As a result, its evaluation policy, though greedy, was based on a less focused, noisier Q-table and performed worse.

 




2.e Training with 10 million episodes improves Q-value coverage and policy quality, allowing the agent to converge more reliably. But for a small game like Tic-Tac-Toe, improvement beyond 1-2 million episodes is minimal due to early saturation of state space.
 
2.f Penalizing ties led the agent to take more aggressive strategies, draws are small 5% While win rate remained high, losses also increased, showing the trade-off between avoiding safe outcomes and playing optimally.

 



Part 3 – Deep and Double Deep Q-Networks

3.a In DQN, the neural network replaces the Q-table by approximating Q-values for all actions given a state. This allows generalization to unseen states and handles large state spaces. However, DQN tends to overestimate Q-values due to using the same network for both action selection and evaluation. DDQN solves this by decoupling the selection (online network) from evaluation (target network), reducing overoptimistic updates.

3.b Implemented in Colab code:
def _build_model(self):
        model = Sequential()                                                         # Build the neural network model
        model.add(Dense(18, input_dim=self.state_size, activation='relu'))     # 2×input size
        model.add(Dense(18, activation='relu'))                           # Another small hidden layer
        model.add(Dense(self.action_size, activation='linear'))   # final 9 values of output
        model.compile(loss='mse', optimizer='adam')

This Architecture is dense enough to learn patterns but not overfit.

3.c 
DDQN showed a modest and inconsistent improvement over DQN, sometimes raising win rates from ~60% to ~80%. While DDQN helps reduce overestimation and stabilize learning, the small size and deterministic nature of Tic Tac Toe limits its impact especially when the opponent plays randomly.

["Best" Network with 18 dense layers as described on 3.b: left – DQN, right – DDQN].





Other architectures which score poorer for reference:



6 then 4 DDQN 










8 then 6 DQN
