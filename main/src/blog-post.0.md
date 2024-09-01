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

## A.I Related Terminology
### Supervised Learning
Supervised learning (SL) is a paradigm in machine learning where input objects (for example, a vector of predictor variables) and a desired output value (also known as a human-labeled supervisory signal) train a model.  
The training data is processed, building a function that maps new data to expected output values.  
An optimal scenario will allow for the algorithm to correctly determine output values for unseen instances.
#### Algorithms
The most widely used learning algorithms are:
- Support-vector machines
- Linear regression
- Logistic regression
- Naive Bayes
- Linear discriminant analysis
- Decision trees
- K-nearest neighbor algorithm
- Neural networks (Multilayer perceptron)
- Similarity learning

### Deep Learning vs Reinforcement Learning
Deep learning and reinforcement learning are both systems that learn autonomously.  
The difference between them is that deep learning is learning from a training set and then applying that learning to a new data set, while reinforcement learning is dynamically learning by adjusting actions based in continuous feedback to maximize a reward.

### LLM
A large language model (LLM) is a computational model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. 
Based on language models, LLMs acquire these abilities by learning statistical relationships from vast amounts of text during a computationally intensive self-supervised and semi-supervised training process.
LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
LLMs are artificial neural networks that utilize the transformer architecture, invented in 2017. The largest and most capable LLMs, as of June 2024, are built with a decoder-only transformer-based architecture, which enables efficient processing and generation of large-scale text data.

## Parametic vs Non-Parametic Learning Algorithm
- Parametic : fixed finite number of parameters ($\theta$) for fitting to the data. So After fitting theta, trainning data is no more needed to make prediction.    
e.g. linear regression
- Non-Parametic : the amount of stuff we need to keep in order to represent the hypothesis $h$ grows linearly with the size of the training set. Also need to have entire training set to make prediction.  
e.g. locally weighted linear regression

### Supervised Learning
As in supervised learning problems, first pick a representation for our hypothesis class (what we are trying to learn) and after that pick a loss function that we will minimize.

### Model and Regression
- Model is a broader concept and encompasses any algorithm that can make predictions.
- Regression specifically refers to a family of methods/models focused on predicting continuous outcome.

#### Linear Model and Regression
Linear model is a broad term that includes any model with a linear relationship between variables, while Linear Regression specifically refers to the task of predicting a continuous outcome using a linear relationship.

#### Logistic Model and Regression
Logistic model is a broader term that refers to any model using a logistic function for classification or other purposes. Logistic regression is a specific statistical method that uses a logistic model to perform binary classification.

### Calibration
Calibration is the degree to which the probabilities predicted by a classification model match the true frequencies of the target classes in a dataset.  
(모형의 예측값이, 실제 확률을 반영하는 것. 예를 들어, X 의 Y1 에 대한 모형의 출력이 0.8이 나왔을 때, 80 % 확률로 Y1 일 것라는 의미를 갖도록 만드는 것입니다.)  
For example, if we make a lot of predictions with a perfectly calibrated binary classification model, and then consider only those for which the model predicted a 70% probability of the positive class, then the model should be correct 70% of the time.  
Similarly, if we only consider the examples for which our model predicted a 10% probability of the positive class, the ground truth will turn out to indeed be positive in one-tenth of the cases.  
A well-calibrated model produces predictions that are closely aligned with the actual outcomes on aggregate.

### Prior Probability & Posterior Probability
#### Prior Probability
A prior probability distribution of an uncertain quantity, often simply called the prior, is its assumed probability distribution before some evidence is taken into account.  
In Bayesian statistics, Bayes' rule prescribes how to update the prior with new information to obtain the posterior probability distribution, which is the conditional distribution of the uncertain quantity given new data. 
#### Posterior Probability
A type of conditional probability that results from updating the prior probability with information summarized by the likelihood via an application of Bayes' rule.  
In the context of Bayesian statistics, it is a conditional distribution over the unobserved random variables, conditioned on the observed random variables. e.g., $p(\theta \mid x,y)$