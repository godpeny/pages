# Generalized Linear Models (GLM)
## Basics
in statistics, a generalized linear model is a flexible generalization of ordinary linear regression. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.  
in a generalized linear model, each outcome $y$ of the dependent variables is assumed to be generated from a particular distribution in an exponential family, a large class of probability distributions that includes the normal, binomial, Poisson and gamma distributions, among others.  

## Exponential Family
an exponential family is a parametric set of probability distributions of a certain form, specified below.  
this special form is chosen formathematical convenience, includingthe enabling of the user to calculateexpectations, covariances usingdifferentiation based on some usefulalgebraic properties, as well as forgenerality, as exponential familiesare in a sense very natural sets ofdistributions to consider. 
$
{\displaystyle \ f_{X}\!\left(x\ {\big |}\ \theta \right)=h(x)\ \exp {\bigl [}\ \eta (\theta )\cdot T(x)-A(\theta )\ {\bigr ]}\ }
$

- sufficient statistic $T(x)$
- naturla parameter $\eta$
- log-partition function $A(\eta)$ : logarithm of a normalization factor, without which ${\displaystyle \ f_{X}\!\left(x\ {\big |}\ \theta \right)}$ would not be a probability distribution:  
$A(\eta) = \log \left( \int_{X} h(x) \exp(\eta(\theta) \cdot T(x)) \, dx \right)$

## 3 Assumptions
GLM consists of three elements below.
1. A particular distribution for modeling $Y$ from among those which are considered exponential families of probability distributions.  
2. A linear predictor $\eta = X\beta$.  
3. A link function $g$ such that $\mathbb{E}(Y \mid X) = \mu = g^{-1}(\eta)$.


## Linear Regression

### Basic

### Margin
### Loss Function(Cost Function)
In mathematical optimization and decision theory, a loss function or cost function (sometimes also called an error function) is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event.  
An optimization problem seeks to minimize a loss function.  
An objective function is either a loss function or its opposite (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized. 
such as,  
$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2.
$  
It is a function that measures, for each value of the $\theta$, how close the $h_\theta(x^{(i)})$ are to the corresponding $y^{(i)}$.  
In other word, choose $\theta$ so as to minimize $J(\theta)$.  

### Gradient Descent
The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the functio(loss function) at the current point, because this is the direction of steepest descent.  
$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta).
$
#### Relation with Loss Function
To minimize cost function $J(\theta)$, let’s start with some “initial guess” for $\theta$, and that repeatedly changes $\theta$ to make $J(\theta)$ smaller, until hopefully we converge to a value of
$\theta$ that minimizes $J(\theta)$.

#### Batch / Stochastic Gradient Descent
- Batch: update parameters (theta) after looking at every example in the entire training set on every iteration. 
Repeat until convergence {  
$
\theta_j := \theta_j + \alpha \sum_{i=1}^{m} \left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right) x_j^{(i)} \quad (\text{for every } j).
$  
}

- Stochastic: update the parameters (theta) according to the gradient of the error with respect to that single training example only.  
$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right) x_j^{(i)} \quad (\text{for every } j).
$

#### Least Mean Square Algorithm (LMS)
LMS algorithm is a stochastic gradient descent method that iteratively updates filter coefficients to minimize the mean square error between the desired and actual signals.  
When error is
$\left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right)$, the magnitude of the update is proportional to the error term.
$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta \left( x^{(i)} \right) \right) x_j^{(i)}.
$
#### Relation with Gradient Descent
$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2} \left( h_\theta(x) - y \right)^2
= 2 \cdot \frac{1}{2} \left( h_\theta(x) - y \right) \cdot \frac{\partial}{\partial \theta_j} \left( h_\theta(x) - y \right)
= \left( h_\theta(x) - y \right) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{i=0}^{n} \theta_i x_i - y \right)
= \left( h_\theta(x) - y \right) x_j.
$  


### Normal Equation

### Probabilistic Interpretation (Maximum Likelihood)
Why least-squre cost function is resonable chocie when faced regression problem?  
Consider hypothesis of regression problem, when $ \epsilon$ is IID(Independently and Identically Distributed) according to Gaussian distribution with mean zero and some variance $\sigma^2$.  
$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}, \ \epsilon^{(i)} \sim \ \mathcal{N}(0, \sigma^2)$   
Since $\epsilon^{(i)} \sim \ \mathcal{N}(0, \sigma^2)$, $
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$  
Which means it can be interpreted as the distribution of $y^{(i)}$ given $x^{(i)}$ parameterized by $\theta$.  
$
p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\left(y^{(i)} - \theta^T x^{(i)}\right)^2}{2\sigma^2}\right).
$  
Given $X$ is design matrix with all elements are $x^{(i)}$ and $\theta$, the distribution of $y^{(i)}$ is $p(\vec{y} | X; \theta)$ (function of $y^{(i)}$).  
We can view this as function of $\theta$ instead of $y^{(i)}$, and this is call likelihood.  
$L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} | X; \theta).$  
$
L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{\left( y^{(i)} - \theta^T x^{(i)} \right)^2}{2\sigma^2} \right) \\
$  
Now, given this probabilistic model relating the $y^{(i)}s$ and the $x^{(i)}s$, reasonable way of choosing best guess of parameter $\theta$ in the principal of maximum likelihood is that choose $\theta$ so as to make the data as high probability as possible.  
I.e., Choose $\theta$ to maximize $L(\theta)$.  
Instead of maximizing $L(\theta)$, we can maximiae log liklihood $log(\theta)$.  
$
\ell(\theta) = \log L(\theta) \\
= \log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{\left( y^{(i)} - \theta^T x^{(i)} \right)^2}{2\sigma^2} \right) \\
= \sum_{i=1}^{m} \log \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{\left( y^{(i)} - \theta^T x^{(i)} \right)^2}{2\sigma^2} \right) \\
= m \log \frac{1}{\sqrt{2\pi\sigma}} - \frac{1}{2\sigma^2} \sum_{i=1}^{m} \left( y^{(i)} - \theta^T x^{(i)} \right)^2.
$  
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

### Probability Interpretation

### Basics

### Newton's Method

## Softmax Regression