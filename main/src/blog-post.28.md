# Reinforcement Learning
Reinforcement Learning(RL) is a branch of machine learning focused on making decisions to maximize cumulative rewards in a given situation. Unlike supervised learning, which relies on a training dataset with predefined answers, RL involves learning through experience.  
In RL, an agent learns to achieve a goal in an uncertain, potentially complex environment by performing actions and receiving feedback through rewards or penalties.

The environment is typically stated in the form of a Markov Decision Process (MDP), as many reinforcement learning algorithms use dynamic programming techniques. 

## Preliminaries

## Finite-State Machine (FSM)
FSM is a mathematical model of computation. It is an abstract machine that can be in exactly one of a finite number of states at any given time. The FSM can change from one state to another in response to some inputs; the change from one state to another is called a transition.  
An FSM is defined by a list of its states, its initial state, and the inputs that trigger each transition. A state is a description of the status of a system that is waiting to execute a transition. A transition is a set of actions to be executed when a condition is fulfilled or when an event is received. 

## Bellman Equation
The Bellman Equation is a recursive formula used in decision-making and reinforcement learning. It shows how the value of being in a certain state depends on the rewards received and the value of future states.   
The Bellman Equation breaks down a complex problem into smaller steps, making it easier to solve. The equation helps find the best way to make decisions when outcomes depend on a series of actions.
$$
V(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s' | s, a) V(s') \right]
$$
 - $V(s)$ : The value of state $s$, which represents the long-term reward of being in that state.
 - $R(s,a)$ : The immediate reward received for taking action $a$ in state $s$.
 - $\gamma$: The discount factor (between 0 and 1) that determines the importance of future rewards compared to immediate rewards.
 - $ P(s' | s, a)$:  The probability of transitioning to state $s′$ from state $s$ by taking action $a$.
 - $\max_{a}$: The optimal action that maximizes the expected value of future rewards.


### Bellman Equation vs Dynamic Programming
 - Dynamic Programming: A method to solve optimization problems using subproblem decomposition and memoization.
 - Bellman Equation: A recursive equation that expresses the value of a state based on the values of successor states.

## Credit Assignment Problem
The credit assignment problem (CAP) is a fundamental challenge in reinforcement learning. It arises when an agent receives a reward for a particular action, but the agent must determine which of its previous actions led to the reward.  
The credit assignment problem refers to the problem of measuring the influence and impact of an action taken by an agent on future rewards. The core aim is to guide the agents to take corrective actions which can maximize the reward.  
Putting it simply, how does algorithm know of all the things that the model did before,  what did it well (which it should do more of) and what did it poorly(which it should do less of).

For example, think of car crashes in building self driving car, chances are the car was doing right before the crash would be brake, but it's not braking that causes the crash and it would be something else that caused crash before crash.
Another example, in chess, when the program lost the game at move 50 due to the blunder(bad move) program amde at move 20 and took another 30 moves before the fates was sealed.

## Markov Decision Processes(MDP)
### Relevance to Reinforcement Learning
MDP is a mathematical framework used to describe an environment in decision making scenarios where outcomes are partly random and partly under the control of a decision-maker(dynamic programing).  
MDP provides a formalism for modeling decision making in situations where outcomes are uncertain, making them essential for Reinforcement Learning.

### Component of MDP
An MDP is defined by a tuple $(S, A, \{ P_{sa} \}, \gamma, R)$ where:

 - States ($S$): A finite set of states representing all possible situations in which the agent can find itself. Each state encapsulates all the relevant information needed to make a decision.
 - Actions ($A$): A finite set of actions available to the agent. At each state, the agent can choose from these actions to influence its environment.
 - Transition Probability ($\{ P_{sa} \}$): A probability function that defines the probability of transitioning from state $s$ to state $s′$ after taking action $a$. This encapsulates the dynamics of the environment. In other words, it gives the distribution over what states($s'$) we will transition to if we take action $a$ in state $s$.
 - Reward Function (R): A reward function $R: S \times A \to \mathbb{R}
$  that provides a scalar reward received after transitioning from state $s$ to state $s′$ due to action $a$. This reward guides the agent towards desirable outcomes.
 - Discount Factor ($\gamma$): A discount factor $\gamma \in [0,1)$ that determines the importance of future rewards. A discount factor close to 1 makes the agent prioritize long-term rewards, while a factor close to 0 makes it focus on immediate rewards.
 - Policy($\pi$): A policy is any function $\pi: S \to A$ mapping from the states to the actions. We say that we are executing some policy $\pi$ if, whenever we are in state $s$, we take action $a = \pi(s)$.

### Basics
The dynamics of an MDP proceeds as follows:  
Starting in some state $s_0$, and choose some action $a_0 \in A$ to take in the MDP. As a result of our choice, the state of the MDP randomly transitions to some successor state $s_1$, drawn according to $s_1 \sim P_{s_0 a_0}$.  
Then, pick another action $a_1$. As a result of this action, the state transitions again, now to some $s_2 \sim P_{s_1 a_1}$. We then pick $a_2$, and so on...  
If you put it into picture, it will be as below.
$$
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \cdots
$$
Upon visiting the sequence of states $s_0, s_1, \cdots$ with actions $a_0,a_1, \cdots$ our total payoff is given by,
$$
R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots
$$
In Reinforcement Learning, the goal is to choose actions over time so as to maximize the expected value of the total payoff:
$$
\mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \right]
$$

Now, let's define the value function for a policy $\pi$.  
$V^{\pi}(s)$ is simply the expected sum of discounted rewards upon starting in state $s$, and taking actions according to a fixed policy $\pi$.

$$
V^{\pi}(s) = \mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \mid s_0 = s, \pi \right]
$$

This can be also expressed as,
$$
V^{\pi}(s) = \mathbb{E} \left[ R(s_0) + \gamma (R(s_1) + \gamma R(s_2) + + \gamma^2 R(s_3)\cdots) \mid s_0 = s, \pi \right]
$$
Now you can see that  value function $V^{\pi}(s)$ satisfies the Bellman equations as below.
$$
V^{\pi}(s) = R(s) + \gamma \sum_{s' \in S} P_{s \pi(s)}(s') V^{\pi}(s').
$$
Where $s$ is current state and $s'$ is the state after one step.
As you can see there are two terms in the equation.
 - $R(s)$: immediate reward that we get rightaway simply for starting in state $s$. 
 - $\sum_{s' \in S} P_{s \pi(s)}(s') V^{\pi}(s')$: the expected sum of future discounted rewards.  


The second term can be interpreted as below which is the expected sum of discounted rewards for starting in state $s′$.  
$$\mathbb{E}_{s' \sim P_{s\pi(s)}} \left[ V^{\pi}(s') \right]$$
Where $s'$ is distributed according $P_{s \pi(s)}(s')$, which is the distribution over where we will end up "after" taking the first action $\pi(s)$ in the MDP from state $s$.  
Therefore the second term gives the expected sum of discounted rewards obtained "after" the first step in the MDP.

### Optimal Value Function and Optimal Policy
From the value functions above $V^{\pi}(s)$, we define the optimal value function is,
$$
V^*(s) = \max_{\pi} V^{\pi}(s).
$$
This is the best possible expected sum of discounted rewards that can be attained using any policy. There is Bellman's Equations version of optimal value function as below.
$$
V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s').
$$
From the above equation first term $R(s)$ is an immediate reward like the value function before.  
The second term is the maximum over all actions $a$ of the expected future sum of discounted rewards we’ll get upon after action $a$. Meaning that it finds the maximum possible expected value over all actions $a \in A$.

There is also optimal policy which is the best action should be taken from state $s$.
$$
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')
$$
Note that $\pi^*(s)$ gives the action $a$ that attains the maximum in the second term in $V^*(s)$ (which is $\sum_{s' \in S} P_{sa}(s') V^*(s')$).

Note that $\pi^*(s)$ has the interesting property that it is the optimal policy for all states $s$.  
Therefore, it is "not" the case that if we were starting in some state $s$ then there’d be some optimal policy for that state, and if we
were starting in some other state $s′$ then there’d be some other policy that’s optimal policy for $s′$.  
The same policy $\pi^*(s)$ attains the maximum in value function for all states $s$. This means that we can use the same policy $\pi^*(s)$ no matter what the initial state of our MDP is.

It is a fact that for every state $s$ and every policy $\pi$, 
$$
V^*(s) = V^{\pi^*}(s) \geq V^{\pi}(s).
$$
Which means that value function for optimal policy $V^{\pi^*}(s)$ is equal to the optimal value function $V^*(s)$ for every state $s$. 

### Value Iteration
One efficient algorithm for solving finite-state MDP is value iteration.  
1. For each state $s$, initialize $V(s) := 0$.
2. Repeat until convergence:  
{  
    For every state, update $V(s) := R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s')$   
}  

In value iteration, the $\pi$ function is not used and instead value function is used.
This can be interpreted as repeatedly trying to update the estimated value function using Bellman Equations.  So more precisely, 
$$
V_{k+1}(s)
\;=\;
R(s)
\;+\;
\gamma\;
\max_{a\in A}
\sum_{s'} P_{s a}(s')\,V_{k}(s').
$$
Note that each new estimate $V_{k+1}(s)$ is defined with the current estimates 
$V_{k}(s)$ on the right-hand side. In other words, you keep applying this Bellman-update operator until the entire vector $V_{k}$ stops changing (converges).  So the recursion is not a separate process for every possible starting state; there is one global value function $V$ that is repeatedly fed back into its own right-hand side.  

Also there are two ways of performing the updates.
 - Synchronous Update: compute the new values for $V(s)$ for every state $s$, and then overwrite all the old values with the new values.
 - Asynchronous Update: loop over the states (in some order) and update the values one at a time.

Under either synchronous or asynchronous updates, value iteration will cause $V$ to converge to $V∗$.
With the optained $V∗$, we can then use find the optimal policy as the equation shown above $\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')$.

### Policy Iteration
Apart from value iteration, there is a second standard algorithm for finding an optimal policy for an MDP which is policy iteration.
1. Initialize $\pi$ randomly.
2. Repeat until convergence:  
{   
    a. Let $V := V^{\pi}$.  
    b. For each state $s$, let $\pi(s) := \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')$   
}

For small MDPs, policy iteration is often very fast and converges with very few iterations.  
However, for MDPs with large state spaces, solving for $V^{\pi}$ explicitly would involve solving a large system of linear equations, and could be difficult. 

### Value Iteration vs Policy Iteration
 - VI: Solve for the $V^{*}$ first, then compute $\pi^*(s)$.
 - PI: Solve Come up with the new policy on every single iteration.   


For small MDPs, policy iteration is often very fast and converges with very few iterations.  
However, for MDPs with large state spaces, solving for $V^{\pi}$$ explicitly would involve solving a large system of linear equations, and could be difficult. In these problems, value iteration may be preferred.

### MDP Model Learning
In many realistic problems, we are not given state transition probabilities and rewards explicitly,
but must instead estimate them from data. ($S$, $A$ and $\gamma$)  
Given this “experience” in the MDP consisting of a number of trials, we can then easily derive the maximum likelihood estimates for the state transition probabilities as below.  
$$
P_{sa}(s') = \frac{\# \text{times took action } a \text{ in state } s \text{ and got to } s'}
{\# \text{times we took action } a \text{ in state } s}
$$
Or, if the ratio above is “0/0”—corresponding to the case of never having taken action $a$in state $s$ before. then we might simply estimate $P_{sa}(s') = \frac{1}{|S|}$.   
Using a similar procedure, if reward function $R$ is unknown, we can also pick our estimate of the expected immediate reward $R(s)$ in state $s$ to be the average reward observed in state $s$.

Having learned a model for the MDP, we can then use either value iteration or policy iteration to solve the MDP using the estimated transition probabilities and rewards.

If putting together model learning ane value iteration(or it can be policy iteation), we can have an algorithm for learning ina an MDP with unknown state transition probabilities.

1. Initialize $\pi$ randomly.
2. Repeat {  
(a) Execute $\pi$ in the MDP for some number of trials.  
(b) Using the accumulated experience in the MDP, update our estimates for $P_{sa}$ (state transition probabilities) (and reward function $R$, if possible).  
(c) Apply value iteration with the estimated state transition probabilities and rewards to get a new estimated value function $V$ .  
(d) Update $\pi$ to be the greedy policy with respect to $V$ .  
}

### Exploration vs Exploitation in RL(MDP) - Exploration–Exploitation dilemma,
Exploitation involves choosing the best option based on current knowledge of the system (which may be incomplete or misleading), while exploration involves trying out new options that may lead to better outcomes in the future at the expense of an exploitation opportunity.  
Finding the optimal balance between these two strategies is a crucial challenge in many decision-making problems whose goal is to maximize long-term benefits.

#### Epsilon-Greedy Algorithm in RL(MDP)
Epsilon-Greedy is a simple method to balance exploration and exploitation by choosing between exploration and exploitation randomly.
The epsilon-greedy, where epsilon refers to the probability of choosing to explore, exploits most of the time with a small chance of exploring.   
For example, at action time $t$, one can take an action $a$ to maximize current reward(greedy with respect to $v$) with probability of $1-\epsilon$, and otherwise you can take an random action with probability of $\epsilon$.

## Continuous State MDPs
We now discuss algorithms for MDPs that may have an infinite number of states, instead of a finite number of states.  
The one simplest way to solve a continuous-state MDP is to discretize the state space, and then to use an algorithm like value iteration or policy iteration, as described previously.  
The alternative method for finding policies in continuous state MDPs, in which we approximate $V^{*}$ directly, without resorting to discretization. 

### Discretization
Discretization is the process of transferring continuous functions, models, variables, and equations into discrete counterparts. 
연속적인 함수, 모델, 변수, 방정식을 이산적인 구성요소로 변환하는 프로세스이다. 이 프로세스는 일반적으로 디지털 컴퓨터에서 수치적 평가 및 구현에 적합하도록 하는 첫 단계로 수행된다.

#### Downside of Discretization in Continuous State MDPs
First, It is a fairly naive representation for value function ($V$) and policy($\pi$).  
Second, because of the curse of dimensionality.  
Suppose $S = \mathbb{R}^{n}$, and we discretize each of the $n$ dimensions of the state into $k$ values. Then the total number of discrete states we have is $k^{n}$.  
This grows exponentially quickly in the dimension of the state space $n$, and thus does not scale well to large problems.  
For example, with a $10$-d state, if we discretize each state variable into $100$ values, we would have $100^{10} = 1020$ discrete states, which is far too many to represent.  
Therefore, discretization very rarely works for problems any higher dimensional than certain number.

### Value Function Approximation
The alternative method for finding policies in continuousstate MDPs, in which we approximate $V^{*}$ directly, without resorting to discretization.  
In linear regression, you can approximate $y$ as a linear function of $x$ as below.
$$
y \approx \theta^T \phi(x), \quad \text{where } \phi(x) = \text{feature of } x =  \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_i^k \end{bmatrix}
$$
Fitted Value Iteration is a model approxmiate $V^*$ of state $s$ as a linear function of the state.
$$
 V^*(s) \approx \theta^T \phi(s)
$$

#### Model(Simulator)
A Model (or simulator) is a black-box that takes as input any (continuous-valued) state $s_t$ and action $a_t$, and outputs a next-state $s_{t+1}$ sampled according to the state transition probabilities $P_{s_ta_t}$.  
You can get a model by using the laws of physics calculate or use open-source software simulator, but we will get a mode by learning from the data collected in the MDP. Let's suppose  execute $m$ trials in which we repeatedly take actions in an MDP, each trial for $T$ timesteps. This can be done picking actions at random, executing some specific policy, or via some other way of choosing actions. Then we can observe $m$ state sequences like the following:
$$
\begin{aligned}
    &s_0^{(1)} \xrightarrow{a_0^{(1)}} s_1^{(1)} \xrightarrow{a_1^{(1)}} s_2^{(1)} \xrightarrow{a_2^{(1)}} \dots \xrightarrow{a_{T-1}^{(1)}} s_T^{(1)} \\
    &s_0^{(2)} \xrightarrow{a_0^{(2)}} s_1^{(2)} \xrightarrow{a_1^{(2)}} s_2^{(2)} \xrightarrow{a_2^{(2)}} \dots \xrightarrow{a_{T-1}^{(2)}} s_T^{(2)} \\
    &\vdots \\
    &s_0^{(m)} \xrightarrow{a_0^{(m)}} s_1^{(m)} \xrightarrow{a_1^{(m)}} s_2^{(m)} \xrightarrow{a_2^{(m)}} \dots \xrightarrow{a_{T-1}^{(m)}} s_T^{(m)}
\end{aligned}
$$
We can then apply a supervised learning algorithm to predict $s_{t+1} $ as a function of $s_t$ and $a_t$.
For example, with the parameter matrices $A$ and $B$, you can choose a linear model of the below form.
$$
s_{t+1} = A s_t + B a_t,
$$
We can estimate them using the data collected from $m$ trials,
$$
\arg\min_{A,B} \sum_{i=1}^{m} \sum_{t=0}^{T-1} \left\| s_{t+1}^{(i)} - \left( A s_t^{(i)} + B a_t^{(i)} \right) \right\|^2.
$$
Note that this corresponds to maximum likelihood estimate.

##### Deterministic Model
Having learned $A$ and $B$, you can build deterministic model, in which given an input $s_t$ and $a_t$, the output $s_{t+1}$ is exactly determined as above equation.
$$
s_{t+1} = A s_t + B a_t,
$$

##### Stochastic Model
 Or you can build a stochastic model, in which $s_{t+1}$ is a random function of the inputs, by modelling it as
$$
s_{t+1} = A s_t + B a_t + \epsilon_t
$$
Where $\epsilon_t$ is a noise term, usually modeled as $\epsilon_t \sim \mathcal{N}(0, \Sigma)$.  
The reason why adding noise is that without noise(= deterministirc model) algorithm might work in the simulator but not in real time. This is because your simulator can never be 100% accurate so adding noise to the simulator can make more roubst policy out of the model. So the odds of generalizaing to real time is much higher.

##### Model-Based RL vs Model-Free RL
 - Model-Based RL: Build a model and train the algorithm in the model. Then, take the policy learned from the model and apply it to the real time.
 - Model-Free RL: Run learning algorithm on the real time directly.

#### Fitted Value Iteration
Let's recall the value iteration from discrete MDP from above section, which repeat below equation until convergence.  
$$
V(s) := R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s')
$$
 - The first term $R(s)$: An immediate reward that we get rightaway simply for starting in state $s$.
 - The second term: The maximum over all actions $a$ of the expected future sum of discounted rewards we’ll get upon after action $a$. (the expected sum of future discounted rewards)
 - Value Function $V(s)$: Expected payoff from the state $s$(expected sum of discounted reward)

However, since now we are now dealing with the continous states rather than discrete states, we have to use integral over states instead of summation. 
$$
V(s) \coloneqq R(s) + \gamma \max_{a} \int_{s'} P_{sa}(s') V(s') \,ds' \\ 
= R(s) + \gamma \max_{a} \mathbb{E}_{s' \sim P_{sa}} \left[ V(s') \right]
$$

The main idea of fitted value iteration is that we are going to approximately carry out this step, over a finite sample of states $s^{(1)}, \cdots, s^{(m)}$.  
What do you mean by approximately carry out? It means that we will use a supervised learning algorithm(linear regression as shown below) to approximate the value function $V(s)$ as a linear or non-linear function of the sampled $m$ states:
$$
V(s) = \theta^T \phi(s).
$$
Where $\phi(s)$ is a feature mapping of state $s$.

##### Algorithm of Fitted Value Iteration
1. Randomly sample $m$ states $s^{(1)}, s^{(2)}, \cdots,  s^{(m)} \in S$. 
2. Initialize $\theta := 0$. 
3. Repeat  
{  
&emsp;For $i = 1, \cdots ,m$, {  
&emsp;&emsp;For each action $a \in A$  {  
&emsp;&emsp;&emsp;Sample $s'_1, \cdots, s'_k ∼ P_{s^{(i)}a}$ (using a model of the MDP)  
&emsp;&emsp;&emsp;Set $q(a) = \frac{1}{k} \sum_{j=1}^{k} R(s^{(i)}) + \gamma V(s'_j)$  
&emsp;&emsp;}  
&emsp;&emsp;Set $y(i) = \max_{a} q(a)$.  
&emsp;&emsp;}  
$\theta := \arg\min_{\theta} \frac{1}{2} \sum_{i=1}^{m} \left( \theta^T \phi(s^{(i)}) - y^{(i)} \right)^2$  
}

From above, $q(a)$ is an estimate of $R(s^{(i)}) + \gamma \mathbb{E}_{s' \sim P_{s^{(i)} a}} \left[ V(s') \right]$. This is because,
$$
R(s^{(i)}) + \gamma \mathbb{E}_{s' \sim P_{s^{(i)} a}} \left[ V(s') \right] = \mathbb{E}_{s' \sim P_{s^{(i)} a}} \left[R(s^{(i)}) + \gamma V(s') \right]
$$
And since we don’t know the exact transition probabilities, we approximate this expectation using sampled next states $s'_j$.
In short, $q(a)$ is an estimate of the expected future due to approximation via sampling.

Similarly, $y(i)$ is an estimate of $R(s^{(i)}) + \gamma \max_{a} \mathbb{E}_{s' \sim P_{s^{(i)} a}} \left[ V(s') \right]$.

Since we defined value function as a linear or non-linear function of the sampled $m$ states, $V(s^{(i)}) =  \theta^T \phi(s^{(i)})$ and we want $V(s^{(i)}) \approx y^{(i)}$, we’ll be using supervised learning (linear regression).  
$$
\theta := \arg\min_{\theta} \frac{1}{2} \sum_{i=1}^{m} \left( \theta^T \phi(s^{(i)}) - y^{(i)} \right)^2
$$
##### Policy of Fitted Value Iteration
Finally, fitted value iteration algorithm outputs $V$ , which is an approximation to $V^{*}$. This implicitly defines our policy because now we have value function $V$.    
Specifically, when our system is in some state s, and we need to choose an action, we would like to choose the action to get optimal policy $\pi^{*},$
$$
\pi^*(s) = \arg\max_{a} \mathbb{E}_{s' \sim P_{s a}} \left[ V(s') \right]
$$
 The process for computing/approximating this is similar to the inner-loop of f itted value iteration, where for each action, we sample $s'_1, \cdots, s'_k ∼ P_{s^{(i)}a}$ to approximate the expectation.

#### In Real Time
You can represent the simulator as, 
$$
s_{t+1} = f(s_t, a_t) + \epsilon_t
$$
Where $f$ is some deterministirc function $f(s_t, a_t) = A s_t + B a_t$ and $\epsilon$ is noise.  
So if you want to use deterministic simulator you have to set nose $\epsilon =0$ and set $k=1$ where $k$ is sampling algorithim from fitted value iteration.  
This is because you are sampling from the expectation over a deterministic distribution, so a single example is sufficient to exactly compute that expectation. (Because no matter how many you sample, result is same in deterministic function)

In real time you don't want the to random sample, because it generates random value and some unlucky value might be very critical.

- Model: Use Stochastic Model.  
$$S_{t+1} = f(S_t, a_t) + \epsilon_t \quad \text{(e.g., } A s_t + B a_t + \epsilon_t \text{)}
$$
- In Real:
Set $\epsilon_t = 0 $ and sampling number $k = 1$.  In other word,  
When in state $s$, pick action, $$\arg\max_{a} V(f(s, a))$$ (Simulator without noise)

During training, add noise to the simulator because it causes the policy you learned to be more robust,
But when deploying in real, it is reasonable to get rid of the noise and set $k=1$ to avoid randomness.