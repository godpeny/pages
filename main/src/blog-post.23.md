# Learning Theory

## Bias and Variance (Trade Off)
![alt text](images/blog23_bias_and_variance.png)
The bias–variance tradeoff describes the relationship between a model's complexity, the accuracy of its predictions, and how well it can make predictions on previously unseen data that were not used to train the model.
As the number of tunable parameters increase in a model, it becomes more flexible, and can better fit a training data set. It is said to have lower error, or $\text{Bias}$.  
However, for more flexible models, there will tend to be greater $\text{Variance}$ to the model fit each time we take a set of samples to create a new training data set.

### Bias
The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).  
= why is prediction staying away from real data.

### Variance
The variance is an error from sensitivity to small fluctuations in the training set. High variance may result from an algorithm modeling the random noise in the training data (overfitting).

## Approximation and Estimation
![alt text](images/blog23_approximation_and_estimation_error.png)

### Bayes Error
$$
\epsilon{(h^*)}
$$
Bayes error rate is the lowest possible error rate for any classifier of a random outcome and is analogous to the irreducible error.

### Estimation Error
$$
\epsilon{(\hat{h})} - \epsilon{(\bar{h})}
$$
The estimation error is the error implied by the fact that the algorithm works with a finite training set that only partially reflects the true distribution of the data.
(limitation do)

### Approximation Error
$$
\epsilon{(h^*)} - \epsilon{(\bar{h})}
$$
The approximation error is the error implied by the choice of function class and is defined as the difference in risk obtained by the best model within the function class and the optimal model.

### Total Error
$$
(\text{Bayes Error})+ (\text{Approximation Error}) + (\text{Estimation Error})
$$

### Relation with Bias and Variance
 - Estimation Error : Estimation Variance + Estimation Bias
 - Variance : Estimation Variance
 - Bais : Estimation Bias + Approximation Error

$$
(\text{Bayes Error})+ (\text{Approximation Error}) + (\text{Estimation Error}) = 
$$
$$
(\text{Bayes Error}) (\text{Bias}) (\text{Variance})
$$

### Fight High Bias & Fight High Variance
 - Increase Hypothesis class $H$ to decrease bias (while increse variance)
 - Increase number of examples to decrease variance. 
 - Regularization to decrease variance.

## Regularization
https://www.geeksforgeeks.org/regularization-in-machine-learning/
https://en.wikipedia.org/wiki/Regularization_(mathematics)

Adding incentive term to make the parameter theta smaller. (when minimizing theta) and make parameter theta bigger when maximizaing theta.


### L1 Regularization
L1 regularization (also called LASSO) leads to sparse models by adding a penalty based on the absolute value of coefficients.
### L2 Regularization
L2 regularization (also called ridge regression) encourages smaller, more evenly distributed weights by adding a penalty based on the square of the coefficients.

## Cross Validation
Cross validation is a technique used in machine learning to evaluate the performance of a model on unseen data. It involves dividing the available data into multiple folds or subsets, using one of these folds as a validation set, and training the model on the remaining folds. This process is repeated multiple times, each time using a different fold as the validation set. Finally, the results from each validation step are averaged to produce a more robust estimate of the model’s performance. Cross validation is an important step in the machine learning process and helps to ensure that the model selected for deployment is robust and generalizes well to new data.

### Purpose of Cross Validation
The main purpose of cross validation is to prevent overfitting, which occurs when a model is trained too well on the training data and performs poorly on new, unseen data. By evaluating the model on multiple validation sets, cross validation provides a more realistic estimate of the model’s generalization performance.

### Holdout Cross Validation (Train, Dev and Test Set)
You train on the training set, evaluate results on the dev set, and test on the test set. So do not test your model on the test set until you have finished handling overfitting.  
In short, 
1. Keep on fitting on train sets.
2. Evaluating and Optimizing the performance of your algorithm on dev sets. (Introducing new features, Choosing model size or regularization parameter...)
3. Want to know how well your algorithm is perforing -> evaluate the model on the test sets.

#### Relation between # of datasets and data split
- If you have to find out small differences in algorithm accuracy, you need large test sets. (e.g., 90.01% vs 90.00%)  
- Choose dev/test sets big enough to make meaningful comparion between different algorithm (90% vs 88%). 
- As the # of example increases, data you send to dev and test sets are shrinking.  
(1,000 example : 60%/20%/20%, 100,000,000,000 example : 98%/1%/1%)

### K-fold Cross Validation
![alt text](images/blog23_k_fold_cross_validation.png)
In K-fold cross-validation, the data set is divided into a number of K-folds and used to assess the model’s ability as new data become available. K represents the number of groups into which the data sample is divided. For example, if you find the k value to be 5, you can call it 5-fold cross-validation. Each fold is used as a test set at some point in the process.

1. Randomly shuffle the dataset.
2. Divide the dataset into k folds
3. For each unique group:
4. Use one fold as test data
5-1. Use remaining groups as training dataset
5-2. Fit model on training set and evaluate on test set
Keep Score 
6. Get accuracy score by applying mean to all the accuracies received for all folds.

### Leave-One-Out Cross Validation
LOOCV(Leave One Out Cross-Validation) is a type of cross-validation approach in which each observation is considered as the validation set and the rest (N-1) observations are considered as the training set.  
In LOOCV, fitting of the model is done and predicting using one observation validation set. Furthermore, repeating this for N times for each observation as the validation set.  
This is a special case of K-fold cross-validation in which the number of folds is the same as the number of observations(K = N). 

## Feature Selection
Feature selection is the process of selecting a subset of relevant features (variables, predictors) for use in model construction.  

Keep on adding feature greedly one at a time to which single feature addition helps improve your algorithm the most until adding more features not hurt performance. Then pick whichever feature subsets allows you to have the best possible performance of dev sets.

Advantages:  
 - simplification of models to make them easier to interpret.
 - shorter training times.
 - to avoid the curse of dimensionality.
 - improve the compatibility of the data with a certain learning model class.
 - to encode inherent symmetries present in the input space.
 
## Empirical Risk Minimizer
Empirical risk minimization is a principle in statistical learning theory which defines a family of learning algorithms based on evaluating performance over a known and fixed dataset.  
The core idea is based on an application of the law of large numbers; more specifically, we cannot know exactly how well a predictive algorithm will work in practice (i.e. the true "risk") because we do not know the true distribution of the data.  
But we can instead estimate and optimize the performance of the algorithm on a known set of training data. (i.e. the empirical risk)  
### Background of ERM
When there is a non-negative real-valued loss function ${\displaystyle L({\hat {y}},y)}$ which measures how different the prediction $\hat{y}$ of a hypothesis is from the true outcome $y$.  
A loss function commonly used in theory is the 0-1 loss function: 
$$
L(\hat{y}, y) =
\begin{cases} 
1 & \text{if } \hat{y} \neq y, \\
0 & \text{if } \hat{y} = y.
\end{cases}
$$
For classification tasks these loss functions can be scoring rules.  
The risk associated with hypothesis $h(x)$ is then defined as the expectation of the loss function:
$$
R(h) = \mathbb{E}[L(h(x), y)] = \int L(h(x), y) \, dP(x, y).
$$
The ultimate goal of a learning algorithm is to find a hypothesis $h^{*}$ among a fixed class of functions $\mathcal {H}$ for which the risk $R(h)$ is minimal:
$$
h^* = \arg\min_{h \in \mathcal{H}} R(h).
$$
### Basics of ERM
In general, the risk $R(h)$cannot be computed because the distribution $P(x,y)$ is unknown to the learning algorithm.   
However, given a sample of iid training data points, we can compute an estimate, called the "empirical risk", by computing the average of the loss function over the training set.  
More formally, computing the expectation with respect to the empirical measure:
$$
R_{\text{emp}}(h) = \frac{1}{n} \sum_{i=1}^n L(h(x_i), y_i).
$$
The empirical risk minimization principle states that the learning algorithm should choose a hypothesis $\hat {h}$ which minimizes the empirical risk over the hypothesis class $\mathcal {H}$:
$$
\hat{h} = \arg\min_{h \in \mathcal{H}} R_{\text{emp}}(h).
$$

### Uniform Convergence
A sequence of functions $f_{n}$ converges uniformly to a limiting function $f$ on a set $E$ as the function domain if, given any arbitrarily small positive number $\epsilon$, a number $N$ can be found such that each of the functions $f_{N},f_{N+1},f_{N+2},\ldots$  differs from $f$ by no more than $\epsilon $ at every point $x$ in $E$.  
Informally speaking, if $f_{n}$ converges to $f$ uniformly, then how quickly the functions $f_{n}$ approach $f$ is "uniform" throughout $E$ in the following sense.  
In order to guarantee that $f_{n}(x)$ differs from $f(x)$ by less than a chosen distance $\epsilon$ we only need to make sure that $n$ is larger than or equal to a certain $N$, which we can find without knowing the value of $x \in E$ in advance.  
In other words, there exists a number $N=N(\epsilon )$ that could depend on $\epsilon$ but is independent of $x$ such that choosing $n\geq N$ will ensure that $|f_{n}(x)-f(x)|<\epsilon$ for all $x\in E$.
![alt text](images/blog23_uniform_convergence.png)
A sequence of functions $f_{n}$ converges uniformly to 
$f$ when for arbitrary small $\epsilon$ there is an index $N$ such that the graph of $f_{n}$ is in the $\epsilon$-tube around $f$ whenever $n\geq N$.

### Union Bound
Union Bound also known as Boole's inequality, says that for any finite or countable set of events, the probability that at least one of the events happens is no greater than the sum of the probabilities of the individual events.  
This inequality provides an upper bound on the probability of occurrence of at least one of a countable number of events in terms of the individual probabilities of the events.
$$
\mathbb{P}\left( \bigcup_{i=1}^\infty A_i \right) \leq \sum_{i=1}^\infty \mathbb{P}(A_i).
$$

### Hoeffding's inequality
Hoeffding's inequality provides an upper bound on the probability that the sum of bounded independent random variables deviates from its expected value by more than a certain amount.

Let $X_{1}+\cdots + X_{n}$ be independent random variables such that $ a_{i}\leq X_{i}\leq b_{i}$ almost surely. Consider the sum of these random variables, $ S_{n}=X_{1}+\cdots + X_{n}$.  
Then Hoeffding's theorem states that, for all $t > 0$ and $\mathbb{E}[S_n]$ is expected value of $S_n$.
$$
\mathbb{P}\left(S_n - \mathbb{E}[S_n] \geq t\right) \leq \exp\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)
$$
$$
\mathbb{P}\left(|S_n - \mathbb{E}[S_n]| \geq t\right) \leq 2 \exp\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)
$$

#### Hoeffding's inequality Generalization
Let $Y_{1},\dots ,Y_{n}$ be independent observations such that 
$\operatorname {E} (Y_{i})=0$ and $a_{i}\leq Y_{i}\leq b_{i}$. Let $\epsilon >0$,  Then, for any $t>0$, 
$$
\mathbb{P}\left(\sum_{i=1}^n Y_i \geq \epsilon\right) \leq \exp\left(-t\epsilon + \sum_{i=1}^n \frac{t^2 (b_i - a_i)^2}{8}\right)
$$

#### Hoeffding's inequality Special Case: Bernoulli Random Variables
Suppose $a_{i}=0$ and $b_{i}=1$ for all $i$. This can occur when $X_{i}$ are independent Bernoulli random variables, though they need not be identically distributed. The set $S_n = X_1 + \cdots + X_n$.
Then we get the inequality for all $t \geq 0$.  
$$
\begin{align*}
\mathbb{P}(S_n - \mathbb{E}[S_n] \geq t) &\leq \exp\left(-\frac{2t^2}{n}\right), \\
\mathbb{P}(|S_n - \mathbb{E}[S_n]| \geq t) &\leq 2 \exp\left(-\frac{2t^2}{n}\right),
\end{align*}
$$
or equivalently, 
$$
\begin{align*}
\mathbb{P}\left(\frac{S_n - \mathbb{E}[S_n]}{n} \geq t\right) &\leq \exp(-2nt^2), \\
\mathbb{P}\left(\left|\frac{S_n - \mathbb{E}[S_n]}{n}\right| \geq t\right) &\leq 2 \exp(-2nt^2).
\end{align*}
$$

## How Uniform Convergence, Union Bound and Hoeffding's inequality related in ERM
### Questions
The Question we want to solve is that.
 1. Can we make formal the bias and variance trade-off.
 2. Even though it's really generalization error that we care about, but most learning algorithms fit their models to the training set. Can we relate error on the training set to generalization error?
 3. Are there conditions under which we can actually prove that learning algorithms will work well?

### Lemmas
 1. Union Bound : The probability of any one of $k$ events happening is at most the sums of the probabilities of the $k$ different events.
 2. Hoeffding Inequality : If we take $\hat{\phi}$ — the average of $m$ Bernoulli($\phi$)random variables — to be our estimate of $\phi$, then the probability of our being far from the true value is small, so long as $m$ is large.

### Preliminaries
We are trying to answer questions using lemmas. Let's assume that
restrict our attention to binary classification in which the labels are $y \in \{0, 1\}$.

$$
S = \left\{ \left(x^{(i)}, y^{(i)}\right); i = 1, \ldots, m \right\}
$$

Given a training set $S$ of size $m$ , where the training examples $(x(i), y(i))$ are drawn iid from some probability distribution $D$. For a hypothesis $h$, we define the training error (also called the empirical risk or empirical error in learning theory) to be below..
$$
\hat{\varepsilon}(h) = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}\{h(x^{(i)}) \neq y^{(i)}\}.
$$
This is just the fraction of training examples that $h$ misclassifies.  

We also define the generalization error to be
$$
\varepsilon(h) = P_{(x,y) \sim \mathcal{D}}(h(x) \neq y).
$$
I.e. this is the probability that, if we now draw a new example $(x, y)$ from
the distribution $D$, $h$ will misclassify it.

Note that we have assumed that the training data was drawn from the same distribution $D$ with which we’re going to evaluate our hypotheses. This is sometimes also referred to as
one of the PAC assumptions.

Consider the setting of linear classification, and let hypothesis $h$ is $h_{\theta}(x) = \mathbb{1}\{\theta^\top x \geq 0\}$. One reasonable way of fitting the parameters is trying to minimize the training error as below.
$$
\hat{\theta} = \arg \min_{\theta} \hat{\varepsilon}(h_{\theta}).
$$
This process is Empirical Risk Minimization (ERM), and the resulting hypothesis output by the learning algorithm is $\hat{h} = h_{\hat{\theta}}$.  

When $H$ is called hypothesis classs and it is the set of
all classifiers. Empirical Risk Minimization can now be thought of as a minimization over the class of functions $H$. In other words the learning algorithm picks the hypothesis based on the fomular below.
$$
\hat{h} = \arg \min_{h \in \mathcal{H}} \hat{\varepsilon}(h).
$$
#### Independent and identically distribute (IID)
A collection of random variables is independent and identically distributed(IID) if each random variable has the same probability distribution as the others and all are mutually independent.  
A random sample can be thought of as a set of objects that are chosen randomly.  
More formally, it is "a sequence of independent, identically distributed (IID) random data points.".  
In other words, the terms random sample and IID are synonymous. 
 - Identically distributed means that there are no overall trends — the distribution does not fluctuate and all items in the sample are taken from the same probability distribution.
 - Independent means that the sample items are all independent events. In other words, they are not connected to each other in any way;[2] knowledge of the value of one variable gives no information about the value of the other and vice versa.

### Finite H
$H$ is just a set of $k$ functions mapping from $X$ to ${0, 1}$, and empirical risk minimization selects $hat{h} to be whichever of these $k$ functions has the smallest training error.

### Infinite H
#### Vapnik-Chervonenkis dimension