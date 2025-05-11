# Statistics
## L1 Norm
The sum of the absolute value of the entries in the vector.
$$
\|\bm{x}\|_1 := \sum_{i=1}^n |x_i|.
$$

## L2 Norm (Euclidean Norm)
The L2 Norm $|x|$ is a vector norm defined for a complex vector (A vector whose elements are complex numbers.)
when 
$$
\mathbf{x} = \begin{bmatrix} 
x_1 \\ 
x_2 \\ 
\vdots \\ 
x_n 
\end{bmatrix}, 
\|\mathbf{x}\| = \sqrt{\sum_{k=1}^n |x_k|^2},
$$

Euclidean Norm is equivalent to the length of the line from the origin to the point and this concept extends to higher dimensions(more than 2-D) as well.

## P-Norm
Let $p\geq 1$ be a real number. The p-norm of vector(also called $\ell^p$-norm) of vector $\bm{x} = (x_1, \dots, x_n)$ is,  
$$
\|\bm{x}\|_p := \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}.
$$

## "Entry-wise" Matrix Norms
Treat $m\times n$  matrix as a vector of size $m\cdot n$ and use one of the familiar vector norms.  
$
\|\mathbf{A}\|_{p,p} = \|\operatorname{vec}(\mathbf{A})\|_p = \left( \sum_{i=1}^m \sum_{j=1}^n \left|a_{ij}\right|^p \right)^{1/p}
$  
The special case, p = 2 is the Frobenius norm, and p = ∞ yields the maximum norm.  
### Lpq Norm
From the original definition matrix, $A$ presents n data points in m-dimensional space.  
Since $(a_{1},\ldots ,a_{n})$ be the columns of matrix 
$\displaystyle L_{2,1}$ norm is the sum of the Euclidean norms of the columns of the matrix:  
$
\|\mathbf{A}\|_{2,1} = \sum_{j=1}^n \|\mathbf{a}_j\|_2 = \sum_{j=1}^n \left( \sum_{i=1}^m \left|a_{ij}\right|^2 \right)^{1/2}
$  
In general, when $p, q \geq 1$,  
$
\|\mathbf{A}\|_{p,q} = \left( \sum_{j=1}^n \left( \sum_{i=1}^m \left|a_{ij}\right|^p \right)^{q/p} \right)^{1/q}.
$
### Frobenius Norm
The Frobenius norm is an extension of the Euclidean norm to $K^{n \times n}$.  
When $p$ = $q$ = 2 for the $L_{p,q}$ it is called the Frobenius norm or the Hilbert–Schmidt norm.  
$
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2} = \sqrt{\text{trace}(A^* A)}.
$  
Loosely speaking, the Frobenius Norm is also equivalent to the Euclidean norm generalised to matrices instead of vectors.

## Gaussian Distribution (= Normal Distribution)
A normal distribution or Gaussian distribution is a type of continuous probability distribution for a real-valued random variable. The general form of its probability density function is,  
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
when the parameter $\mu$ is the mean or expectation of the distribution (and also its median and mode), while the parameter 
$\sigma ^{2}$ is the variance.  
Their importance is partly due to the central limit theorem. It states that, under some conditions, the average of many samples (observations) of a random variable with finite mean and variance is itself a random variable—whose distribution converges to a normal distribution as the number of samples increases. 

## Joint Probability
A joint probability is the probability of events A and B happening at the same time.  
$P(A, B) = P(A ∩ B)$ : likelihood of events occurring together at the same point in time.

## Probability vs Likelihood
- In Deep Learning
  - same function can be interpreted in two ways: as a function of data given parameters, or as a function of parameters given data. 
  - probability : function of data given parameters. (parameters are fixed and data is varying)
  - likelihood : function of parameters given data. (data is fixed and parameters are varying)
- In Statistics
  - probability : chance that a particular outcome occurs based on the values of parameters in a model. 
    - (확률 분포가 고정된 상태에서, 관측되는 사건이 변화될 때 확률)
  - likelihood : how probable a particular set of observations is, given a set of model parameters. = likelihood is used to estimate the parameters of a model given a set of observations.
    - (관측된 사건이 고정된 상태에서, 확률 분포가 변화될 때(=확률 분포를 모를 때 = 가정할 때) 확률)
  - In discrete case : probability = likelihood
  - In PDF
    - probability = area under the curve.
    - likelihood = y-axis value.

## Likelihood Function vs Posterior Distribution
 - Likelihood Function: the probability of the evidence($X$) given the parameters($\theta$) which is, $P(X|\theta)$ = 어떤 모델에서 해당 데이터(관측값)이 나올 확률.
 - Posterior Distribution: the probability of the parameters given the evidence which is, $P(\theta|X)$ = 사건이 발생한 후(관측이 진행된 후) 그 사건이 특정 모델에서 발생했을 확률. 
$$
  p(\theta | x) = \frac{p(x | \theta)}{p(x)} p(\theta)
$$
e.g., 
```
바닥에 떨어진 머리카락의 길이(z)를 보고 그 머리카락이 남자 것인지 여자 것인지 성별(x)을 판단하는  문제:

 - ML(Maximum Likelihood) 방법: ML 방법은 남자에게서 그러한 머리카락이 나올 확률 p(z|남)과 여자에게서 그러한 머리카락이 나올 확률 p(z|여)을 비교해서 가장 확률이 큰, 즉 likelihood가 가장 큰 클래스(성별)를 선택하는 방법이다.

 - MAP(Maximum A Posteriori) 방법: MAP 방법은 z라는 머리카락이 발견되었는데 그것이 남자것일 확률 p(남|z), 그것이 여자것일 확률 p(여|z)를 비교해서 둘 중 큰 값을 갖는 클래스(성별)를 선택하는 방법이다. 즉, 사후확률(posterior prabability)를 최대화시키는 방법으로서 MAP에서 사후확률을 계산할 때 베이즈 정리가 이용된다.

ML과 MAP 차이는 남녀의 성비를 고려하면 명확해진다. 
ML은 남녀의 성비는 완전히 무시하고 순수하게 남자중에서 해당 길이의 머리카락을 가질 확률, 여자중에서 해당 길이의 머리카락을 가질 확률만을 비교하는 것이다. 반면에 MAP는 각각의 성에서 해당 머리카락이 나올 확률 뿐만 아니라 남녀의 성비까지 고려하여 최종 클래스를 결정하는 방법이다.
```
$$
p(\text{여} | z) = \frac{p(\text{여} \land z)}{p(z)} = \frac{p(\text{여} \land z)}{p(\text{여} \land z) + p(\text{남} \land z)} = \frac{p(z | \text{여}) p(\text{여})}{p(z | \text{여}) p(\text{여}) + p(z | \text{남}) p(\text{남})}
$$

## Maximum Likelihood Esstimation
A method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of observing the data.  
In other word, when modeling a set of observations as a random sample from an unknown joint probability distribution which is expressed in terms of a set of parameters.  
The goal of maximum likelihood estimation is to determine the parameters for which the observed data have the highest joint probability. We write the parameters governing the joint distribution as a vector $\theta = \begin{bmatrix} \theta_1, \theta_2, \dots, \theta_k \end{bmatrix}^T$ so that this distribution falls within a parametric family $\{ f(\cdot; \theta) \mid \theta \in \Theta \}$, where $\Theta$ is called the parameter space, a finite-dimensional subset of Euclidean space.  

 - Likelihood: 확률 분포가 가정된 상황에서 관측값이 가지는 확률값. 확률 분포가 가정된 상황에서 관측값이 가지는 확률값.
 - MLE: 관측되는 데이터들을 가장 잘 모델링하는 확률분포의 parameter를 찾는 알고리즘.

In order to maximize likelihood $L(\theta)$,  it is sometimes easier to work with $\ell(\theta) := \log L(\theta)$
instead.  
Since the logarithm function is monotonically increasing, it is clear
that the value of $\theta$ where the maximum of $L(\theta)$ occurs is necessarily the value
of $\theta$ where the maximum of $\ell(\theta)$ occurs, and that the converse is also true.
That is,
$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\theta) \iff \hat{\theta}_{\text{MLE}} = \arg\max_\theta \log L(\theta).
$$
To find the maximum, we take the derivative of the log-likelihood function with respect to the parameters ($=\ell'(\theta))$ and set it equal to zero, which gives us the likelihood equations, which are necessary conditions for finding the maximum.  
Solving these equations provides the parameter values that maximize the likelihood function

## Maximum a posteriori estimation (MAP)
사후확률(posterior probability)을 사전확률(prior probability)과 likelihood를 이용해서 계산할 수 있도록 해 주는 확률 변환식.  
In Bayesian statistics, a maximum a posteriori probability (MAP) estimate is an estimate of an unknown quantity, that equals the mode of the posterior distribution.  
It is closely related to the method of maximum likelihood (ML) estimation, but employs an augmented optimization objective which incorporates a prior distribution (that quantifies the additional information available through prior knowledge of a related event) over the quantity one wants to estimate.
Observe that the MAP estimate of 
${\displaystyle \theta }$ coincides with the ML estimate when the prior 
${\displaystyle g}$ is uniform.  
When $p(\theta|x,y)$ is posterior distribution and $p(y|x,\theta)$ is likelihood and $p(\theta)$ is prior destribution, 
$$
\begin{align*}
\theta_{\mathrm{MAP}} & = \arg \max_\theta p(\theta \ \vert \ x, y) \\
                      & = \arg \max_\theta \frac{p(y \ \vert \ x, \theta) \ p(x, \theta)}{p(x, y)} \\
                      & = \arg \max_\theta \frac{p(y \ \vert \ x, \theta) \ p(\theta \ \vert \ x) \ p(x)}{p(x, y)} \\
                      & = \arg \max_\theta \frac{p(y \ \vert \ x, \theta) \ p(\theta) \ p(x)}{p(x, y)} \\
                      & = \arg \max_\theta p(y \ \vert \ x, \theta) \ p(\theta)
\end{align*}
$$

## Probability Distribution
Probability Distribution is a mathematical description of the probabilities of events, subsets of the sample space. The sample space, is the set of all possible outcomes of a random phenomenon being observed.  
The sample space may be any set: a set of real numbers, a set of descriptive labels, a set of vectors, a set of arbitrary non-numerical values, etc. For example, the sample space of a coin flip could be  $\omega = \{ "heads", "tails" \}$ .


https://en.wikipedia.org/wiki/Probability_distribution

### Terminology of Probability Distribution

### Discrete Probability Distributions

### Absolutely Continuous Probability Distributions
An absolutely continuous probability distribution is a probability distribution on the real numbers with uncountably many possible values, such as a whole interval in the real line, and where the probability of any event can be expressed as an integral.  
More precisely, a real random variable $X$ has an absolutely continuous probability distribution if there is a function $f:\mathbb {R} \to [0,\infty]$ such that for each interval $ I=[a,b]\subset \mathbb {R}$, the probability of $X$ belonging to $I$ is given by the integral of $f$ over $I$:
$$
{\displaystyle P\left(a\leq X\leq b\right)=\int _{a}^{b}f(x)\,dx.}
$$
This is the definition of a probability density function, so that absolutely continuous probability distributions are exactly those with a probability density function.

### Properties of Probability Distribution
1. Under the distribution $P$, if $X$ is a random variable with distribution $P$, then for any function $f$, expectation is as below.
$$
\mathbb{E}_{X\sim P}[f(X)]
\;=\;
\sum_{x \in \mathcal{X}}\,P(x)\,f(x).
$$
2. Any probability distribution must sum (or integrate) to 1 over its entire support. 
$$
\sum_{x\in \mathcal{X}}\,P(x) \;=\; 1.
$$
3. Intuitively, if you try to sum a function over all possible $x$, that indicator acts like a filter that leaves only the single term at $x^{(i)}$.
$$
\sum_{x}\,\mathbf{1}\{x^{(i)} = x\}\,\log P_\theta(x)
\;=\;
\log P_\theta\!\bigl(x^{(i)}\bigr).
$$
So you can think of it as “selecting” precisely the $\log P_\theta(x)$ value at the observed data point $x^{(i)}$.



## Cumulative Distribution Function(CDF) vs Probability Density Function(PDF) vs Probability Mass Function(PMF)
### PDF: continuous random variable
![alt text](images/blog5_pdf.png)
#### Necessary Conditions for PDF
 - $f(x) \geq 0, \, \forall \, x \in \mathbb{R}$
 - $f(x)$ should be piecewise continuous.
 - $\int_{-\infty}^{\infty} f(x) \, dx = 1$

#### Properties of PDF
 - $\Pr[a \leq X \leq b] = \int_a^b f_X(x) \, dx$
 - $E(X) := \int_{-\infty}^{\infty} x f(x) \, dx$
 - $\text{Var}(X) := \int_{-\infty}^{\infty} [x - E(X)]^2 f(x) \, dx$
 - if $F_X$ is the cumulative distribution function(CDF) of $X$,  
$F_X(x) = \int_{-\infty}^x f_X(u) \, du,$
and (if $f_X$ is continuous at $x$) $f_X(x) = \frac{d}{dx} F_X(x)$.
 - the probability density function of a continuous random variable over a single value is zero.  
 $P(X = a) = P(a \leq X \leq a) = \int_a^a f(x) \, dx = 0$

## Expected Value vs Mean vs Average
 - average and mean : mathematically, average and mean are same. So basic formulas used to calculate average and mean are also the same. But the difference between them lies in context in which they are use. The term average is used to estimate an approximate value of a given data in general purpose. However, the use of word in “mean” is specifically used in the context of statistics. In other words, mean is specifically used to represent the average of the statististical data.
(average == mean)
 - mean vs expected value : mean is typically used when we want to calculate the average value of a given sample. This represents the average value of raw data that we’ve already collected. However, expected value is used when we want to calculate the mean of a probability distribution. This represents the average value we expect to occur before collecting any data. 
 In other words, expected value is generalization of the weighted average. Informally, the expected value is the arithmetic mean of the possible values a random variable can take, weighted by the probability of those outcomes. 

### PMF : discrete random variable
A function that gives the probability that a discrete random variable is exactly equal to some value. Sometimes it is also known as the discrete probability density function.  
#### Properties of PMF
- $p : \mathbb{R} \to [0, 1], \ \ p_X(x) = P(X = x) \\$
- $\sum_x p_X(x) = 1, \ \ p_X(x) \geq 0$

### CDF : the probability that the random variable X that takes on a value less than or equal to x
The cumulative distribution function of a real-valued random variable $X$ is the function given by,  
$
F_X(x) = P(X \leq x)
$
#### Properties of CDF
when $f(x)$ is probability density function and $F(x)$ is cumulative distribution function.  

- ${P}(a < X \leq b) = F_X(b) - F_X(a) \\$
- $f(x) = \frac{dF(x)}{dx} \\$
- $F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt \\$ 
- ${P}(X = b) = F_X(b) - \lim_{x \to b^-} F_X(x)$

## Joint Probability Mass Function
The joint probability mass function of two discrete random variables $X$ and $Y$ is defined as
$$
P_{XY}(x, y) = P(X = x, Y = y).
$$

## Mean Square Error (MSE)
The 'mean squared error (MSE)' or 'mean squared deviation (MSD) of an estimator' (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors.  
The average squared difference between the estimated values and the actual value.  
### Definition
If a vector of $n$ predictions is generated from a sample of $n$ data points on all variables, and $Y$ 
is the vector of observed values of the variable being predicted, with $\hat {Y}$ being the predicted values (e.g. as from a least-squares fit), then the within-sample MSE of the predictor is computed as
$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \hat{Y}_i \right)^2
$  
since the mean is $\left( \frac{1}{n} \sum_{i=1}^{n} \right)$ and error is $\left( Y_i - \hat{Y}_i \right)^2 = \left( e_i \right)^2$
In matrix calculation,   
$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( e_i \right)^2 = \frac{1}{n} \mathbf{e}^T \mathbf{e}
$  
where $e_i$ is $Y_i - \hat{Y}_i$ and $\mathbf{e}$ is $n \times 1$ column vector.

## Point estimation
In statistics, point estimation involves the use of sample data to calculate a single value (known as a point estimate since it identifies a point in some parameter space) which is to serve as a "best guess" or "best estimate" of an unknown population parameter (for example, the population mean).

## Understanding of $p(x|z)$ (Conditional Probability)
The notation "p(x|z)" means "the probability of $x$ given $z$," and 
When we say "given" in $p(x|z)$, it means that outcome $x$ occurs given that another event $z$ has already occurred and the probability of $x$ happening depends on that of $z$ happening.

1. Conditional Probability  
   - Considering the probability of $x$ under the condition that we know the value of $z$.
   - Representing how the probability of $x$ changes when we have information about $z$.
2. Fixed or Known Information  
   - Treating $z$ as known or fixed information.
   - looking at the probability distribution of $x$ in the subset of cases where $z$ takes on a specific value.
3. Updated Knowledge  
   - The probability of $x$ is updated based on the information provided by z.
   - It reflects how our understanding of $x$ changes when we learn about $z$.
4. Restricted Sample Space:
   - Restricting our consideration to only those outcomes where $z$ has a particular value.

For example, when $z$ is a random vector as below,
$$
\mathbf{z} = \begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_n
\end{bmatrix}
$$
Expected value of the random vector $z$ given $z$ can be interpreted as Computing the expected value of the random vector $z$ given that we already know its value as following.
$$
\mathbb{E} \left[ \mathbf{z} \mid \mathbf{z} \right] = \mathbf{z}
$$

Let's suppose Suppose $z$ is a random variable (scalar) when $z=5$ given. Then the conditional expectation of $z$ given $z$ ($p(z|z)$) is simply the value of $z$ itself.
$$
E[Z∣Z=5]=5
$$

If z is not a scalar but vector, then 
$$
\mathbf{z} = \begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_n
\end{bmatrix}
$$
The expected value of each component $z_i$ given that we know $z_i$ is simply $z_i$ itself.
$$
\mathbb{E} \left[ \begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_n
\end{bmatrix} \mid \mathbf{\mathbf{z} = \begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_n
\end{bmatrix}} \right] = \begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_n
\end{bmatrix}
$$

Let's see other example too.
$$
\mathbb{E}[x \mid z] = \mathbb{E}[\mu + \Lambda z + \epsilon \mid z] = \mathbb{E}[\mu \mid z] + \mathbb{E}[\Lambda z \mid z] + \mathbb{E}[\epsilon \mid z] \\
= \mu + \Lambda z + 0 = \mu + \Lambda z \quad \text{(since } \mu \text{ and } \Lambda z \text{ are constant given } z)
$$

$$
\operatorname{Cov}(x \mid z) = \operatorname{Cov}(\mu + \Lambda z + \epsilon \mid z) (\text{since } \mu + \Lambda \text{ are constants given } z)\\
= \operatorname{Cov}(\epsilon \mid z) = \Psi \\

\therefore \quad x \mid z \sim \mathcal{N}(\mu + \Lambda z, \Psi).
$$

## Joint Probability Distribution
The probability distribution of all possible pairs of outputs of two random variables that are defined on the same probability space. 

When dealing with continuous random variables, things are a bit different. The notation $p(x) = p(x_1, x_2)$ refers to the joint probability density function (pdf), not the probability of an intersection.  
It is the pdf that describes how likely the pair $(x_1, x_2)$ is.

For example,  
Each of two urns contains twice as many red balls as blue balls, and no others, and one ball is randomly selected from each urn, with the two draws independent of each other. Let $A$ and $B$ be discrete random variables associated with the outcomes of the draw from the first urn and second urn respectively.  
The probability of drawing a red ball from either of the urns is ⁠$\frac{2}{3}$ and the probability of drawing a blue ball is ⁠⁠$\frac{1}{3}$⁠. The joint probability distribution is presented in the following table.

$$
\begin{array}{|c|c|c|c|}
\hline
 & A = \text{Red} & A = \text{Blue} & P(B) \\
\hline
B = \text{Red} & \left( \frac{2}{3} \right) \left( \frac{2}{3} \right) = \frac{4}{9} & \left( \frac{1}{3} \right) \left( \frac{2}{3} \right) = \frac{2}{9} & \frac{4}{9} + \frac{2}{9} = \frac{2}{3} \\
\hline
B = \text{Blue} & \left( \frac{2}{3} \right) \left( \frac{1}{3} \right) = \frac{2}{9} & \left( \frac{1}{3} \right) \left( \frac{1}{3} \right) = \frac{1}{9} & \frac{2}{9} + \frac{1}{9} = \frac{1}{3} \\
\hline
P(A) & \frac{4}{9} + \frac{2}{9} = \frac{2}{3} & \frac{2}{9} + \frac{1}{9} = \frac{1}{3} &  \\
\hline
\end{array}
$$

Each of the four inner cells shows the probability of a particular combination of results from the two draws. These probabilities are the joint distribution. In any one cell the probability of a particular combination occurring is (since the draws are independent) the product of the probability of the specified result for $A$ and the probability of the specified result for $B$.  
The probabilities in these four cells sum to $1$, as with all probability distributions.

### Marginal Probability Distribution
It is important to distinguish between the joint probability distribution of $X$ and $Y$ and the probability distribution of each variable individually.  

The individual probability distribution of a random variable is referred to as its marginal probability distribution. In general, the marginal probability distribution of $X$ can be determined from the joint probability distribution of $X$ and other random variables(e.g., $Y$).

From above example, The final row and the final column give the marginal probability distribution for $A$ and the marginal probability distribution for $B$ respectively.  
For example, for $A$ the first of these cells gives the sum of the probabilities for A being red, regardless of which possibility for $B$ in the column above the cell occurs, as ⁠⁠$\frac{2}{3}$⁠.  
Thus the marginal probability distribution for $A$ gives $A$'s probabilities unconditional on $B$, in a margin of the table.

If the joint probability density function of random variable $X$ and $Y$ is $f_{X,Y}(x,y)$ , the marginal probability density function of $X$ and $Y$, which defines the marginal distribution is as following.
$$
f_X(x) = \int f_{X,Y}(x, y) \, dy \\
f_Y(y) = \int f_{X,Y}(x, y) \, dx
$$


## Normalization and Regularization
- Normalization: A data preprocessing technique that adjusts the values of features to a common scale, typically between 0 and 1, without distorting the differences in the range of values. 
- Regularization: A technique used to prevent overfitting in a model by adding a penalty term to the loss function. Regularization helps the model generalize better by preventing it from becoming too complex.  

There are two common types of regularization: L1-norm (Lasso) and L2-norm (Ridge Regression). Both of these methods add a penalty term to the loss function, which encourages the model to use simpler fitting functions and reduces the magnitude of the model parameters.

### Whitening transformation
https://en.wikipedia.org/wiki/Whitening_transformation

### Standard Normalization
In mathematical statistics, a random variable $$ is standardized by subtracting its expected value $\operatorname {E} [X]$ and dividing the difference by its standard deviation $\sigma (X)={\sqrt {\operatorname {Var} (X)}}$,
$$
Z={X-\operatorname {E} [X] \over \sigma (X)}
$$

For standard normalized value $z$, mean of $z$ is 0 and variance is 1.

#### Mean of Standard Normalized value $z$
$$
\mu_z = \mathbb{E}[z] = \frac{1}{n} \sum_{i=1}^{n} z_i = \frac{1}{n} \sum_{i=1}^{n} \frac{x_i - \mu_x}{\sigma_x}
$$
Split the right hand side as follow,
$$
\mu_z = \frac{1}{\sigma_x} \left( \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_x) \right)
$$
Since we know that $\mu_x = \frac{1}{n} \sum_{i=1}^{n} x_i$, 
$$
\mu_z = \frac{1}{\sigma_x} \left( \frac{1}{n} \sum_{i=1}^{n} x_i \right ) - \frac{1}{\sigma_x} \mu_x \\
\mu_z = \frac{1}{\sigma_x} ( \mu_x -  \mu_x) = 0 \\
\therefore \mu_z = 0
$$

#### Variance of Standard Normalized value $z$
$$
\operatorname{Var}(z) = \mathbb{E}[z^2] = \frac{1}{n} \sum_{i=1}^{n} z_i^2 = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \mu_x}{\sigma_x} \right)^2
$$
Factoring out the $\sigma_x$ in the right hand side we get,
$$
\operatorname{Var}(z) = \frac{1}{\sigma_x^2} \cdot \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_x)^2
$$
Since we know that $\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_x)^2 = \sigma_x^2$, 
$$
\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_x)^2 = \sigma_x^2 \\
\therefore \operatorname{Var}(z) = \frac{1}{\sigma_x^2} \cdot \sigma_x^2 = 1
$$