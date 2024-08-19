# Generalized Linear Models (GLM)
## Basics
A generalized linear model is a flexible generalization of ordinary linear regression. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.  
In a generalized linear model, each outcome $y$ of the dependent variables is assumed to be generated from a particular distribution in an exponential family, a large class of probability distributions that includes the normal, Bernoulli, binomial, Poisson and gamma distributions, among others.  

## Exponential Family
An exponential family is a parametric(finite number of parameters) set of probability distributions of a certain form.   
This special form is chosen for mathematical convenience, including the enabling of the user to calculate expectations, covariances using differentiation based on some useful algebraic properties, as well as forgenerality, as exponential familiesare in a sense very natural sets ofdistributions to consider. 
$$
{\displaystyle \ f_{X}\!\left(x\ {\big |}\ \theta \right)=h(x)\ \exp {\bigl [}\ \eta (\theta )\cdot T(x)-A(\theta )\ {\bigr ]}\ }
$$

- sufficient statistic $T(x)$
- naturla parameter $\eta$
- log-partition function $A(\eta)$ : logarithm of a normalization factor, without which ${\displaystyle \ f_{X}\!\left(x\ {\big |}\ \theta \right)}$ would not be a probability distribution:  
$A(\eta) = \log \left( \int_{X} h(x) \exp(\eta(\theta) \cdot T(x)) \, dx \right)$

## 3 Assumptions
GLM consists of three elements below.
1. A particular distribution for modeling $Y$ from among those which are considered exponential families of probability distributions.  
2. A linear predictor $\eta = X\beta$.  
3. A link function $g$ such that $\mathbb{E}(Y \mid X) = \mu = g^{-1}(\eta)$.

### Bernoulli Distribution as GLM
$$
\begin{align*}
p(y; \phi) &= \phi^y (1 - \phi)^{1 - y} \\
&= \exp\left(y \log \phi + (1 - y) \log (1 - \phi)\right) \\
&= \exp\left(\left(\log\left(\frac{\phi}{1 - \phi}\right)\right) y + \log(1 - \phi)\right).
\end{align*}
$$  
$$
\begin{align*}
\eta &= \log\left(\frac{\phi}{1 - \phi}\right) \\ 
T(y) &= y \\
a(\eta) &= -\log(1 - \phi) \\
        &= \log(1 + e^{\eta}) \\
b(y) &= 1 \\
\end{align*}
$$

### Gausian Distribution as GLM
$$
\begin{align*}
p(y; \mu) &= \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} (y - \mu)^2 \right) \\
&= \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} y^2 \right) \cdot \exp\left( \mu y - \frac{1}{2} \mu^2 \right)
\end{align*}
$$  
$$
\begin{align*}
\eta &= \mu \\
T(y) &= y \\
a(\eta) &= \frac{\mu^2}{2} \\
       &= \frac{\eta^2}{2} \\
b(y) &= \left(\frac{1}{\sqrt{2\pi}}\right) \exp\left(-\frac{y^2}{2}\right).
\end{align*}
$$

## Linear Regression
### Basic
In Machine Learning, linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data.   
 - Simple Linear Regression: Only one independent variable($x$).  
 (only one $x$ and one $y$ variable.)
 - Multiple Linear Regression: Relationship between two or more variables($x$) and a response($y$) by fitting a linear equation to observed data.  
 (one $y$ and more than one $x$ variables. e.g.,$y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \epsilon$)
 - Univariate Linear Regression : Target variable depends on only one independent variable.  
 (one $x$ and one $y$) 
 - Multivariate Linear Regression : More than one predictor and more than one response.  
(more than one $x$ and more than one $y$ variable.)  
 $\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{\Xi}$  
$$
\begin{pmatrix}
y_{11} \ y_{12} \ \cdots \ y_{1p} \\
y_{21} \ y_{22} \ \cdots \ y_{2p} \\
y_{31} \ y_{32} \ \cdots \ y_{3p} \\
\vdots \ \vdots \ \ddots \ \vdots \\
y_{n1} \ y_{n2} \ \cdots \ y_{np}
\end{pmatrix}
=
\begin{pmatrix}
1 \ x_{11} \ x_{12} \ \cdots \ x_{1q} \\
1 \ x_{21} \ x_{22} \ \cdots \ x_{2q} \\
1 \ x_{31} \ x_{32} \ \cdots \ x_{3q} \\
\vdots \ \vdots \ \vdots \ \ddots \ \vdots \\
1 \ x_{n1} \ x_{n2} \ \cdots \ x_{nq}
\end{pmatrix}
\begin{pmatrix}
\beta_{01} \ \beta_{02} \ \cdots \ \beta_{0p} \\
\beta_{11} \ \beta_{12} \ \cdots \ \beta_{1p} \\
\beta_{21} \ \beta_{22} \ \cdots \ \beta_{2p} \\
\vdots \ \vdots \ \ddots \ \vdots \\
\beta_{q1} \ \beta_{q2} \ \cdots \ \beta_{qp}  
\end{pmatrix} + 
\begin{pmatrix}
\epsilon_{11} \ \epsilon_{12} \ \cdots \ \epsilon_{1p} \\
\epsilon_{21} \ \epsilon_{22} \ \cdots \ \epsilon_{2p} \\
\epsilon_{31} \ \epsilon_{32} \ \cdots \ \epsilon_{3p} \\
\vdots \ \vdots \ \ddots \ \vdots \\
\epsilon_{n1} \ \epsilon_{n2} \ \cdots \ \epsilon_{np}
\end{pmatrix}
$$

### Margin
### Loss Function(Cost Function)
In mathematical optimization and decision theory, a loss function or cost function (sometimes also called an error function) is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event.  
An optimization problem seeks to minimize a loss function.  
An objective function is either a loss function or its opposite (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized. 
such as,  
$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2.
$$  
It is a function that measures, for each value of the $\theta$, how close the $h_\theta(x^{(i)})$ are to the corresponding $y^{(i)}$.  
In other word, choose $\theta$ so as to minimize $J(\theta)$.  

### Gradient Descent
The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the functio(loss function) at the current point, because this is the direction of steepest descent.  
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta).
$$
#### Relation with Loss Function
To minimize cost function $J(\theta)$, let’s start with some “initial guess” for $\theta$, and that repeatedly changes $\theta$ to make $J(\theta)$ smaller, until hopefully we converge to a value of
$\theta$ that minimizes $J(\theta)$.

#### Batch / Stochastic Gradient Descent
- Batch: update parameters (theta) after looking at every example in the entire training set on every iteration. 
Repeat until convergence {  
$$
\theta_j := \theta_j + \alpha \sum_{i=1}^{m} \left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right) x_j^{(i)} \quad (\text{for every } j).
$$  
}

- Stochastic: update the parameters (theta) according to the gradient of the error with respect to that single training example only.  
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right) x_j^{(i)} \quad (\text{for every } j).
$$

#### Least Mean Square Algorithm (LMS)
LMS algorithm is a stochastic gradient descent method that iteratively updates filter coefficients to minimize the mean square error between the desired and actual signals.  
When error is
$\left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right)$, the magnitude of the update is proportional to the error term.
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right) x_j^{(i)}.
$$
#### Relation with Gradient Descent
$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2} \left( h_\theta(x) - y \right)^2
= 2 \cdot \frac{1}{2} \left( h_\theta(x) - y \right) \cdot \frac{\partial}{\partial \theta_j} \left( h_\theta(x) - y \right)
= \left( h_\theta(x) - y \right) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{i=0}^{n} \theta_i x_i - y \right)
= \left( h_\theta(x) - y \right) x_j.
$$  


### Normal Equation

### Probabilistic Interpretation (Maximum Likelihood)
Why least-squre cost function is resonable chocie when faced regression problem?  
Consider hypothesis of regression problem, when $ \epsilon$ is IID(Independently and Identically Distributed) according to Gaussian distribution with mean zero and some variance $\sigma^2$.  
$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}, \ \epsilon^{(i)} \sim \ \mathcal{N}(0, \sigma^2)$   
Since $\epsilon^{(i)} \sim \ \mathcal{N}(0, \sigma^2)$, $
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$  
Which means it can be interpreted as the distribution of $y^{(i)}$ given $x^{(i)}$ parameterized by $\theta$.  
$$
p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\left(y^{(i)} - \theta^T x^{(i)}\right)^2}{2\sigma^2}\right).
$$  
Given $X$ is design matrix with all elements are $x^{(i)}$ and $\theta$, the distribution of $y^{(i)}$ is $p(\vec{y} | X; \theta)$ (function of $y^{(i)}$).  
We can view this as function of $\theta$ instead of $y^{(i)}$, and this is call likelihood.  
$L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} | X; \theta).$  
$$
L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{\left( y^{(i)} - \theta^T x^{(i)} \right)^2}{2\sigma^2} \right) \\
$$  
Now, given this probabilistic model relating the $y^{(i)}s$ and the $x^{(i)}s$, reasonable way of choosing best guess of parameter $\theta$ in the principal of maximum likelihood is that choose $\theta$ so as to make the data as high probability as possible.  
I.e., Choose $\theta$ to maximize $L(\theta)$.  
Instead of maximizing $L(\theta)$, we can maximiae log liklihood $log(\theta)$.  
$$
\ell(\theta) = \log L(\theta) \\
= \log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{\left( y^{(i)} - \theta^T x^{(i)} \right)^2}{2\sigma^2} \right) \\
= \sum_{i=1}^{m} \log \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{\left( y^{(i)} - \theta^T x^{(i)} \right)^2}{2\sigma^2} \right) \\
= m \log \frac{1}{\sqrt{2\pi\sigma}} - \frac{1}{2\sigma^2} \sum_{i=1}^{m} \left( y^{(i)} - \theta^T x^{(i)} \right)^2.
$$  
Above derivation, we can find the fact that maximizing $log(\theta)$ is same as minimizing 
$\sum_{i=1}^{m} \left( y^{(i)} - \theta^T x^{(i)} \right)^2$, which is original loss function $J(\theta)$.

### Locally Weighted Linear Regression (LWR)
Rather than learning a fixed set of parameters as is done in ordinary linear regression, parameters $\theta$ are computed individually for each query point $x$.
$
J(\theta) = \sum_{i=1}^{m} w^{(i)} \left( \theta^T x^{(i)} - y^{(i)} \right)^2
$  
When original linear regression:  
 - fit $\theta$ to minimize $\sum_{i=1}^{m} w^{(i)} \left( \theta^T x^{(i)} - y^{(i)} \right)^2$
 - predict with $ \theta^T x^{(i)}$.  

LWR:
 - fit $\theta$ to minimize $\sum_{i=1}^{m} w^{(i)} \left( \theta^T x^{(i)} - y^{(i)} \right)^2$
 - predict with $ \theta^T x^{(i)}$.  

#### Weight
Weight $w^{(i)}s$ are non negative value. A standard choice the weight is  
$
w^{(i)} = \exp \left( -\frac{\left(x^{(i)} - x\right)^2}{2\tau^2} \right)
$  


## Logistic Regression
### Basics
Logistic regression is a supervised machine learning algorithm used for classification tasks,  
where the goal is to predict the probability that an instance belongs to a given class or not.   
Logistic regression is a statistical algorithm which analyze the relationship between two data factors. 

### Logistic Function (Sigmoid Function)
when $h_{\theta}(x)$ is hypothesis, and $\theta^{T} x = \theta_{0} + \sum_{j=1}^{n} \theta_{j} x_{j}$,  
$$
h_{\theta}(x) = g(\theta^{T} x) = \frac{1}{1 + e^{-\theta^{T} x}},
$$  
It is also useful to check derivate of sigmoid function which is:  
$$
g'(z) = \frac{d}{dz} \left(\frac{1}{1 + e^{-z}}\right) \\
= \frac{1}{(1 + e^{-z})^2} \left(e^{-z}\right) \\
= \frac{1}{(1 + e^{-z})} \cdot \left(1 - \frac{1}{(1 + e^{-z})}\right) \\
= g(z)(1 - g(z)).
$$

### Bernoulli Distribution
The discrete probability distribution of a random variable which takes the value 1 with probability $p$ and the value 0 with probability $ q=1-p$.  
$$
f(k; p) =
\begin{cases} 
p & \text{if } k = 1, \\
q = 1 - p & \text{if } k = 0.
\end{cases}
$$

### Classification Model using Logistic Regression
When assuming,  
$$
P(y = 1 \mid x; \theta) = h_\theta(x) \\
P(y = 0 \mid x; \theta) = 1 - h_\theta(x) \\
=  \\
p(y \mid x; \theta) = \left( h_\theta(x) \right)^y \left( 1 - h_\theta(x) \right)^{1-y}
$$   
Likelihood of the parameter is,  
$$
L(\theta) = p(\vec{y} \mid X; \theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta) \\
= \prod_{i=1}^{m} \left( h_{\theta}(x^{(i)}) \right)^{y^{(i)}} \left( 1 - h_{\theta}(x^{(i)}) \right)^{1 - y^{(i)}}.
$$  
Log likelihood of the parameter is (for convience),  
$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_{\theta}(x^{(i)}) + (1 - y^{(i)}) \log \left( 1 - h_{\theta}(x^{(i)}) \right) \right].
$$  
In order to maximize the likelihood, use gradient ascent method similar to linear regression problem.(the positive
rather than negative sign in the update formula, since we’re maximizing, rather than minimizing)  
So the update will be:  
$$
\theta := \theta + \alpha \nabla_{\theta} \ell(\theta)
$$  
Taking derivatives to log likelihood is:  
$$
\frac{\partial}{\partial \theta_j} \ell(\theta) = \left( \frac{y}{g(\theta^T x)} - \frac{(1-y)}{1 - g(\theta^T x)} \right) \cdot \frac{\partial}{\partial \theta_j} g(\theta^T x) \\
= \left( \frac{y}{g(\theta^T x)} - \frac{(1-y)}{1 - g(\theta^T x)} \right) g(\theta^T x) \left( 1 - g(\theta^T x) \right) \cdot x_j \\
= \left( y(1 - g(\theta^T x)) - (1-y)g(\theta^T x) \right) \cdot x_j \\
= \left( y - h_{\theta}(x) \right) \cdot x_j.
$$  
Therefore stochastic gradient ascent rule is:  
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_{\theta}(x^{(i)}) \right) x_j^{(i)}
$$  

### Why Logistic Regression linear model?
Logistic regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. ($\theta^T x = \theta_{0} x_{0} + \theta_{1} x_{1} \cdots \theta_{m} x_{m}$)  
In other words, the output cannot depend on the product (or quotient, etc.) of its parameters.(example of non linear: $ \theta_{1} x_{1} \times  \theta_{2} x_{2} ...$)  
 
### Newton's Method

## Softmax Regression