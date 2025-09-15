# Stub

## General Terminology
### Variables vs Parameters
- Variables : values that are input to a model.
  - i.e. input features in a neural network. ( = x)
- Parameters : values that are learned by the model during training.
  - i.e. weights, biases in a neural network. ( = W, b) 
  - theta : set of parameters in a model.

### ArgNax / ArgMin
- Arguments of maxima / minima.
- input points at which the function output value is maximized / minimized.

### Indicator function (Characteristic function)
 - Characteristic Function in Convex Analysis is a convex function that indicates the membership (or non-membership) of a given element in that set. Let $X$ be a set, and let $A$ be a subset of $X$. The characteristic function of $A$ is, 
 $$
 \chi_A(x) :=
\begin{cases} 
0, & x \in A; \\
+\infty, & x \notin A.
\end{cases}
$$
 - Indicator Function is a function that maps elements of the subset to one, and all other elements to zero. That is, if $A$ is a subset of some set $X$, then, 
 $$
 1_A : X \to \{0, 1\}
 $$

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

### Necessity and Sufficiency
In logic and mathematics, necessity and sufficiency are terms used to describe a conditional or implicational relationship between two statements.  
For example, when conditional sentence "If $P$ then $Q$", 
$$P \Rightarrow Q"$$
 $Q$ is necessity for $P$ because the truth of $Q$ is guranteed by $P$.  
Similarly $P$ is sufficient for $Q$, because $P$ being true always implies that $Q$ is true. (But $P$ not being true does not always imply that $Q$ is not true)

### Polynomial
In mathematics, a polynomial is a mathematical expression consisting of indeterminates (also called variables) and coefficients, that involves only the operations of addition, subtraction, multiplication and exponentiation to nonnegative integer powers, and has a finite number of terms.

### Coefficient
In mathematics, a coefficient is a multiplicative factor involved in some term of a polynomial, a series, or any other type of expression. It may be a number without units, in which case it is known as a numerical factor.[1] It may also be a constant with units of measurement, in which it is known as a constant multiplier.

### Variable
In mathematics, a variable is a symbol, typically a letter, that holds a place for constants, often numbers. One say colloquially that the variable represents or denotes the object, and that the object is the value of the variable.  
A variable may represent a unspecified number that remains fixed during the resolution of a problem; in which case, it is often called a parameter. A variable may denote an unknown number that has to be determined

### Math Symbols
 - Logical NOT: It negates the condition that follows.
$$
\neg
$$
 - There exists: There exists hypothesis $h$ in the hypothesis class $\mathcal{{H}}$.
$$
\exists h \in \mathcal{H}
$$
 - For all : For all hypothesis $h$ in the hypothesis class $\mathcal{{H}}$.
$$
\forall h \in \mathcal{H}
$$
 -  there does not exist: the probability that there does not exist some element or condition satisfying a particular property. 
 $$
 \neg \exists
 $$

### Coefficient and Variable in Polynomial
$$
2X^2 - 3X + 2
$$
From above polynomial, coefficients are $2$, $-3$, $2$, and variable is $X$.

$$
aX^2 + bX + c = 0
$$
From above polynomial, parameters are $a,b,c$ and variable is $X$.

### Radon's theorem
### Convex Hull
### Additive Structure
In Machine Learning, additive structure refers to a functional form where the output is the sum of contributions from individual components, 
often represented as:  
$$
f(x) = \sum_{i=1}^{k} f_i(x_i).
$$
Each feature $x_i$ contributes independently to the output $f(x)$.  
Common examples include linear models and generalized additive models (GAMs).
It emphasizes how the model combines inputs additively, focusing on simplicity and interpretability.

In Mathematics, additive structure can describe systems where addition (or an analogous operation) plays a defining role, such as vector spaces, rings, or groups. It highlights how elements interact under the operation of addition.

### Interchanging the Order of Summation
$$
\text{If } \sum_{j=1}^\infty \sum_{k=1}^\infty |a_{jk}| < \infty, \text{ then } \sum_{j=1}^\infty \sum_{k=1}^\infty a_{jk} = \sum_{k=1}^\infty \sum_{j=1}^\infty a_{jk}.
$$

### Orientation
The orientation also knwon as attitude, bearing, direction, or angular position of an object – such as a line, plane or rigid body – is part of the description of how it is placed in the space it occupies. It refers to the imaginary rotation that is needed to move the object from a reference placement to its current placement.  

A rotation may not be enough to reach the current placement, in which case it may be necessary to add an imaginary translation to change the object's position (or linear position). The position and orientation together fully describe how the object is placed in space. 

### Angular Orientation
Angular Orientation($\displaystyle {\vec {\omega }}$) is a pseudovector representation of how the angular position or orientation of an object changes with time.  
For example, how quickly an object rotates (spins or revolves) around an axis of rotation and how fast the axis itself changes direction.

### Piece-Wise Function
In mathematics, a piecewise function (also called a hybrid function, or a function defined by cases) is a function whose domain is partitioned into several intervals ("subdomains") on which the function may be defined differently. As an example, consider the piecewise definition of the absolute value function,
$$
\displaystyle |x|={\begin{cases}-x,&{\text{if }}x<0\\+x,&{\text{if }}x\geq 0.\end{cases}}
$$
(Piece-Wise: with respect to a number of discrete intervals, sets, or pieces)

See also below two examples.
<img src="images/blog0_piece-wise_function.png" alt="Markov Chain" width="400"/>   
The left piece-wise linear function is defined as below.
$$
{\displaystyle f(x)=\left\{{\begin{array}{lll}-3-x&{\text{if}}&x\leq -3\\x+3&{\text{if}}&-3\leq x\leq 0\\3-2x&{\text{if}}&0\leq x\leq 3\\0.5x-4.5&{\text{if}}&3\leq x\\\end{array}}\right.}
$$
The right function is defined as below.
$$
{\displaystyle f(x)=\min(1,x^{2})} 
$$

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

### Parametic vs Non-Parametic Learning Algorithm
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

#### Calibration vs Accuracy
Accuracy measures the percentage of correct predictions made by the model, while calibration measures the alignment between the predicted probabilities and the actual likelihood of the predicted events.

### Prior Probability & Posterior Probability
#### Prior Probability
A prior probability distribution of an uncertain quantity, often simply called the prior, is its assumed probability distribution before some evidence is taken into account.  
In Bayesian statistics, Bayes' rule prescribes how to update the prior with new information to obtain the posterior probability distribution, which is the conditional distribution of the uncertain quantity given new data. 
#### Posterior Probability
A type of conditional probability that results from updating the prior probability with information summarized by the likelihood via an application of Bayes' rule.  
In the context of Bayesian statistics, it is a conditional distribution over the unobserved random variables, conditioned on the observed random variables. e.g., $p(\theta \mid x,y)$

### Error 
Errors refer to the discrepancies between the predicted output of a neural network model and the actual or desired output. These errors are used to compute the loss or cost function, which measures how well the model is performing on a given task. The goal of training a deep learning model is to minimize these errors or the associated loss function.

### Training error(Empirical error) vs Generalization error
Training error is measured on training data - the data used to construct the model.  
Generalization error is the error expected on new cases, and is usually estimated by measuring the error on a test data set, which is not used during model construction.

### Affine Layer
An affine layer, also called a fully connected layer or dense layer, is a layer in which the input signal of the neuron is multiplied by the weight, summed, and biased. An affine layer can be a layer of an artificial neural network in which all contained nodes connect to all nodes of the subsequent layer.  
It is a type of layer where each input is connected to each output by a learnable weight. Affine layers are commonly used in both traditional neural networks and deep learning models to transform input features into outputs that the network can use for prediction or classification tasks.  

### Covariate Shift
Covariate shift is a specific type of dataset shift often encountered in machine learning. It is when the distribution of input data shifts between the training environment and live environment. Although the input distribution may change, the output distribution or labels remain the same. Covariate shift is also known as covariate drift, and is a very common issue encountered in machine learning. Models are usually trained in offline or local environments on a sample of labelled training data. It’s not unusual for the distribution of inputs in a live and dynamic environment to be different from the controlled training environment.  

### Logit
In machine learning, logit is the vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function.  
If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

### Model vs Embedding
<b> Model </b>
- Everything needed to re-create a Keras / TF model:
- Layer graph / architecture
- All trainable weights (including any embedding layer)
- (Optionally) optimizer state, loss, metrics
- .h5, .keras, sometimes .pb
```
model_shakespeare_kiank_350_epoch.h5
└── architecture JSON
└── layer weights  (Embedding, LSTM, Dense …)
└── optimizer state (Adam, SGD … if present)
```
```python
from keras.models import load_model
model = load_model("model_shakespeare.h5", compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy")
```

<b> Embedding </b>
- Only a lookup table that maps tokens → vectors (e.g. GloVe, word2vec, fastText). No layer definitions, no other weights
- .txt, .vec, .bin, sometimes .npy, .pkl
```
the        0.418 0.249 …  (100 floats)
to         0.680 0.540 …
```
```python
embedding_matrix = np.loadtxt("glove.6B.100d.txt", skiprows=1)
embedding_layer = keras.layers.Embedding(
       input_dim=vocab_size,
       output_dim=100,
       weights=[embedding_matrix],
       trainable=False)
```

### Markov Model
Markov model is a stochastic model used to model pseudo-randomly changing systems. It is assumed that future states depend only on the current state, not on the events that occurred before it. The simplest Markov model is the Markov chain. 

#### Markov Chain
A Markov chain is a way to describe a system that moves between different situations called "states", where the chain assumes the probability of being in a particular state at the next step depends solely on the current state. 

For example, let'x consider two-state Markov chain below.
<img src="images/blog0_markov_chain.png" alt="Markov Chain" width="200"/>   

If in state A:
- Stays in A: probability 0.6
- Moves to E: probability 0.4  

If in state E:
- Moves to A: probability 0.7
- Stays in E: probability 0.3

Also note that a Markov chain can be illustrated as a directed graph, where nodes represent the states (A, E), arrows indicate possible transitions and the numbers on arrows show transition probabilities.

### Auto-Regressive Model
Autoregressive modeling is a machine learning technique most commonly used for time series analysis and forecasting that uses one or more values from previous time steps in a time series to create a regression.  
The Auto-Regressive(AR) Model is defined as follws.
$$
{\displaystyle X_{t}=\sum _{i=1}^{p}\varphi _{i}X_{t-i}+\varepsilon _{t}}
$$
where $\varphi _{1},\ldots ,\varphi _{p}$ are the parameters of the model, and $\varepsilon _{t}$ is white noise.

### Tips for reading papers
Compile list of paper (including blogs and medium posts) and skipping around the list.
Steady learning, Not short burst.

<b> Reading 1 Paper</b>
1. Title, Abstract, Figures
2. Intro, Conclusition and skim rest (skip related works)
3. Read but skip the math
4. Whole things but skip parts that don't make sense.

<b> To have good understanding of paper </b>
Make sure you can have answer to questions below.
- What did authors try to accomplish?
- What were the key elements of the approach?
- What can you use yourself?
- What other references do you want to follow?

<b> Sources of papers </b>
- Twitter
- Reddit - ML Subreddit
- NIPS/ICMC/ICLR

<b> Math and Code </b>
- Try to rederive the math or code from the scratch.

<b> Machine Learning Engineer </b>
![alt text](images/blog0_machine_learning_engineer.png)

- Have broad understanding of many different topics in AI Machine Learning. While also have very deep understanding in at least one area. 
- To deepen the understanding, work on project, open-source contribution, research or job.