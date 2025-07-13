# Deep-Reinforcement-Learning
Deep Reinforcement learning project with hands-on experience implementing advanced version of the classic Tic Tac Toe game
Project 4 – TIC-TAC-TOE with Reinforcement Learning 
Student Name: Yair Lahad
Submission Date: 30/06/2025

Part 1 – Monte Carlo

1.a Solution on code.
1.b Learning Rate: low LR Is stable and converges. Higher LR converges fast for less epochs (20k) yet unstable with more epochs.
<img width="792" height="467" alt="image" src="https://github.com/user-attachments/assets/6841be4d-503f-4e65-a6ea-02e781659310" />

Discount Factor is currently not used in Monte Carlo implementation but in general its effect:
<img width="858" height="429" alt="image" src="https://github.com/user-attachments/assets/177be3c6-a159-4572-8ea4-0eb18fc1702b" />

 
Epsilon effect on wins/loss: could not directly understand strategy usage. A proper method to learn about epsilon is analyzing final performance.
<img width="944" height="527" alt="image" src="https://github.com/user-attachments/assets/328af484-1344-4728-a150-4451b6d4adec" />


Episode number impact on the agent overall performance: helps until converges around 40k episodes
 <img width="975" height="475" alt="image" src="https://github.com/user-attachments/assets/5e29ddb1-a820-4060-a426-6a7573b6f75c" />

1.c From the analysis on section 1.b and from parameter analysis done in code I choose the following:
	1. High Learning Rate = 0.5
	2. High Epsilon = 0.7 as seen in 1.b as the optimized value.
<img width="975" height="442" alt="image" src="https://github.com/user-attachments/assets/2e5e2fd0-e9d4-4093-a478-37497f87f9c8" />

 <img width="975" height="454" alt="image" src="https://github.com/user-attachments/assets/1b1f7f09-4a4f-420f-9a28-584761ba754b" />

As seen on the Win/Loss/Draw rates, applying those parameters increases the overall performance of the Agent each time.


Part 2 – Q Learning 

2.On code. This is one baseline for evaluation. 
<img width="975" height="454" alt="image" src="https://github.com/user-attachments/assets/f0f63e2d-a838-4027-9ca4-3ac9d37a1978" />

2.b Agent learning haven't improved after changing the learning rate to 0.1. It lost more than baseline stats. Meaning it is less stable with worse convergence.

 <img width="582" height="471" alt="image" src="https://github.com/user-attachments/assets/e2571694-ddcf-40b2-8b23-a8e7c6d32bba" />

2.c
For 0.9 discount factor: Agent learns better by valuing future rewards less extremely than baseline.
even though training stats show worse win rate and higher exploration rate, agent wins more on evaluation, fit for short games like Tic-Tac-Toe.
<img width="559" height="460" alt="image" src="https://github.com/user-attachments/assets/fe004900-bbd6-4a43-89d8-d42b0e2d78f6" />
 
For 0.5 discount factor: Agent learns poorly by not valuing future rewards enough.
 <img width="544" height="435" alt="image" src="https://github.com/user-attachments/assets/415b429d-a7ab-4b91-a2e3-3f6319f65ea4" />

2.d The ε = 1 agent learned passively from random episodes without ever exploiting its Q-values, so it lacked consistent reinforcement of successful strategies. As a result, its evaluation policy, though greedy, was based on a less focused, noisier Q-table and performed worse.
 <img width="537" height="430" alt="image" src="https://github.com/user-attachments/assets/bb485a93-1e30-45ca-8650-bc3cd6601a47" />


2.e Training with 10 million episodes improves Q-value coverage and policy quality, allowing the agent to converge more reliably. But for a small game like Tic-Tac-Toe, improvement beyond 1-2 million episodes is minimal due to early saturation of state space.
<img width="519" height="430" alt="image" src="https://github.com/user-attachments/assets/af80e89c-9f1d-4461-b22d-dbb7d50ce31a" />

2.f Penalizing ties led the agent to take more aggressive strategies, draws are small 5% While win rate remained high, losses also increased, showing the trade-off between avoiding safe outcomes and playing optimally.

 <img width="519" height="430" alt="image" src="https://github.com/user-attachments/assets/a902b3d2-3f03-4745-abde-3b48166787e4" />


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

<img width="523" height="424" alt="image" src="https://github.com/user-attachments/assets/b6aad4ab-5336-4591-8c88-69535b690e6c" />
<img width="528" height="427" alt="image" src="https://github.com/user-attachments/assets/57ba91c2-9263-4ade-99c9-d07e31a06dbb" />


Other architectures which score poorer for reference:

<img width="474" height="379" alt="image" src="https://github.com/user-attachments/assets/5eaf257e-fc05-4810-80f2-711a5972ffa1" />

6 then 4 DDQN 

<img width="468" height="385" alt="image" src="https://github.com/user-attachments/assets/8a2fa01e-d21f-4e55-a2c9-2f6fb5ba48a6" />

8 then 6 DQN
