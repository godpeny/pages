# Stub

## General Terminology
### Variables vs Parameters
- Variables : values that are input to a model.
  - i.e. input features in a neural network. ( = x)
- Parameters : values that are learned by the model during training.
  - i.e. weights, biases in a neural network. ( = W, b) 
  - theta : set of parameters in a model.

### Train, Dev, Test Set
train on the training set, evaluate results on the dev set, and test on the test set.
so do not test your model on the test set until you have finished handling overfitting.

### ArgNax / ArgMin
- Arguments of maxima / minima.
- input points at which the function output value is maximized / minimized.

### Indicator function (Characteristic function)
https://en.wikipedia.org/wiki/Characteristic_function_(convex_analysis)

### Coordinate Vector
coordinate vector is a representation of a vector as an ordered list of numbers (a tuple) that describes the vector in terms of a particular ordered basis.  e.g.  
$
B = \{b_1, b_2, \dots, b_n\} \\
v = \alpha_1 b_1 + \alpha_2 b_2 + \dots + \alpha_n b_n \\
[v]_B = (\alpha_1, \alpha_2, \dots, \alpha_n)
$
coordinate vector of $v$ relative to B is $[v]_B$

### Soft vs Hard Constraints (Constrained Optimization)
- Soft Constraints
Some variable values that are penalized in the objective function if, and based on the extent that, the conditions on the variables are not satisfied.
- Hard constraints
Set conditions for the variables that are required to be satisfied.

### Deep Learning vs Reinforcement Learning
Deep learning and reinforcement learning are both systems that learn autonomously.  
The difference between them is that deep learning is learning from a training set and then applying that learning to a new data set, while reinforcement learning is dynamically learning by adjusting actions based in continuous feedback to maximize a reward.

### Reinforcement Learning
https://en.wikipedia.org/wiki/Reinforcement_learning
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture16-guest.pdf


### Deep Reinforcement Learning (deep RL)
https://huggingface.co/learn/deep-rl-course/en/unit0/introduction

### Graph Neural Network (GNN)
https://en.wikipedia.org/wiki/Graph_neural_network

### LLM
A large language model (LLM) is a computational model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. 
Based on language models, LLMs acquire these abilities by learning statistical relationships from vast amounts of text during a computationally intensive self-supervised and semi-supervised training process.
LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
LLMs are artificial neural networks that utilize the transformer architecture, invented in 2017. The largest and most capable LLMs, as of June 2024, are built with a decoder-only transformer-based architecture, which enables efficient processing and generation of large-scale text data.