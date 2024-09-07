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
- canonical response function(distribution's mean as a function of the natural parameter) : $g(\eta) = \mathbb{E}[T(y);\eta]$  
(the canonical response function for the
Gaussian family is just the identify function; and the canonical response function for the Bernoulli is the logistic function)  
- canonical link function: $g^{-1}$

### Link Function and Response Function
![alt text](images/blog8_link_response_relationship.png)
$$
\eta = g\left(\gamma'(\theta)\right) \\
\theta = \gamma'^{-1}\left(g^{-1}(\eta)\right)
$$
 - $\gamma'$ is known function.
 - $g$ is link function.

#### Canonical
$g$ is canonical link function if the function connects $\eta$, $\theta$ and $\mu$.
$$
\gamma'^{-1} \circ g^{-1} = \left(g \circ \gamma'\right)^{-1} = I \\
$$
Same as,
$$
\eta = \theta
$$


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
such as,  
$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2.
$$  
An objective function is either a loss function or its opposite (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized.    
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
### Jacobian with Matrix
Given a training set let's define the design matrix $X$ which contains the training examples’ input values in its rows:  
$$
X = \begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\vdots \\
(x^{(m)})^T
\end{bmatrix}.
$$
And let $\vec{y}$ be the $m$-dimensional vector containing all the target values from the training set:  
$$
\vec{y} = 
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{bmatrix}.
$$
Since $(x^{(1)})^T\theta = h_\theta(x^{(1)}) - y^{(1)}$, we can verify,  
$$
X\theta - \vec{y} = 
\begin{bmatrix}
(x^{(1)})^T\theta \\
\vdots \\
(x^{(m)})^T\theta
\end{bmatrix}
- 
\begin{bmatrix}
y^{(1)} \\
\vdots \\
y^{(m)}
\end{bmatrix}
= 
\begin{bmatrix}
h_\theta(x^{(1)}) - y^{(1)} \\
\vdots \\
h_\theta(x^{(m)}) - y^{(m)}
\end{bmatrix}.
$$
Using the fact that $z^T z = \sum_i z_i^2$ for vector $z$,  
$$
\frac{1}{2} (X \theta - \vec{y})^T (X \theta - \vec{y}) = \frac{1}{2} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 = J(\theta)
$$
To minimize, $J(\theta)$, let's find derivatives with respect to $\theta$ using Normal Equation,  
$$
\nabla_\theta J(\theta) = \nabla_\theta \frac{1}{2} (X \theta - \vec{y})^T (X \theta - \vec{y}) = X^T X \theta - X^T \vec{y}
$$
To minimize $J(\theta)$, we set its derivatives to zero, and obtain the
normal equations:
$$
X^T X \theta =  X^T \vec{y} \\
 \theta = X^T X \theta - X^T \vec{y}
 $$

### Probabilistic Interpretation (Maximum Likelihood)
Why least-squre cost function is resonable chocie when faced regression problem?  
Consider hypothesis of regression problem, when $ \epsilon$ is IID(Independently and Identically Distributed) according to Gaussian distribution with mean zero and some variance $\sigma^2$.  
$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}, \ \epsilon^{(i)} \sim \ \mathcal{N}(0, \sigma^2)$   
Since $\epsilon^{(i)} \sim \ \mathcal{N}(0, \sigma^2)$, $
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$  
Which means it can be interpreted as the distribution of $y^{(i)}$ given $x^{(i)}$ parameterized by $\theta$ as $y^{(i)} \mid x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$.  
Which is:  
$$
p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\left(y^{(i)} - \theta^T x^{(i)}\right)^2}{2\sigma^2}\right).
$$  
Note that we should not condition on $\theta$
$p(y^{(i)} | x^{(i)}; \theta)$, since $\theta$ is not a random variable. (Remeber that Gaussian distribution is a type of continuous probability distribution for a real-valued random variable)  
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
Logistic regression is a supervised machine learning algorithm used for classification tasks, where the goal is to predict the probability that an instance belongs to a given class or not.   
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

### Binary Classification Model in Logistic Regression (label: 1,0)
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
\frac{\partial}{\partial \theta_j} \ell(\theta) = \sum_{i=1}^{m} \left( \frac{y^{(i)}}{g(\theta^T x^{(i)})} - \frac{(1-y^{(i)})}{1 - g(\theta^T x^{(i)})} \right) \cdot \frac{\partial}{\partial \theta_j} g(\theta^T x^{(i)}) \\
= \sum_{i=1}^{m} \left( \frac{y^{(i)}}{g(\theta^T x^{(i)})} - \frac{(1-y^{(i)})}{1 - g(\theta^T x^{(i)})} \right) g(\theta^T x^{(i)}) \left( 1 - g(\theta^T x^{(i)}) \right) \cdot x_j^{(i)} \\
= \sum_{i=1}^{m} \left( y^{(i)}(1 - g(\theta^T x^{(i)})) - (1-y)g(\theta^T x^{(i)}) \right) \cdot x_j^{(i)} \\
= \sum_{i=1}^{m} \left( y^{(i)} - h_{\theta}(x^{(i)}) \right) \cdot x_j^{(i)}.
$$  
Therefore stochastic gradient ascent rule is:  
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_{\theta}(x^{(i)}) \right) x_j^{(i)}
$$  

### Why Logistic Regression linear model?
Logistic regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. ($\theta^T x = \theta_{0} x_{0} + \theta_{1} x_{1} \cdots \theta_{m} x_{m}$)  
In other words, the output cannot depend on the product (or quotient, etc.) of its parameters.(example of non linear: $ \theta_{1} x_{1} \times  \theta_{2} x_{2} ...$)  

### Binary Classification Model using Logistic Regression (label: 1,-1)
#### Margin of Binary Classification 
In binary Classification problems, it is often convenient to use a hypothesis class of the form $h_{\theta}(x) = \theta^T x$ and when presented with a new example $x$, we classify it
as positive or negative depending on the sign of $\theta^T x$.  
$$
\text{sign}(h_{\theta}(x)) = \text{sign}(\theta^T x) \quad \text{where} \quad \text{sign}(t) =
\begin{cases} 
1 & \text{if } t > 0 \\
0 & \text{if } t = 0 \\
-1 & \text{if } t < 0.
\end{cases}
$$
Where we say that $y = 1$ if the example is a member of the
positive class and $y = −1$ if the example is a member of the negative class, $y \in \{-1, +1\}$.  
Then, the hypothesis $h_{\theta}(x)$ with parameter
vector $\theta$ classifies a particular example ($X$, $y$) correctly if,  
$$
\text{sign}(\theta^T x) = y \quad \text{or equivalently} \quad y\theta^T x > 0.
$$  
$y\theta^Tx$ is called margin.  

#### Loss Function for Binary Classification
Choose some loss function so that
for our training data, makes the margin $y^{(i)}\theta^Tx^{(i)}$
very large for each training example.  
When $\varphi$ is loss function with zero-one range and $z$ is $y\theta^Tx$ :  
$$
\varphi_{zo}(z) = \varphi_{zo}(y\theta^Tx) = 
\begin{cases} 
1 & \text{if } z \leq 0, \\
0 & \text{if } z > 0.
\end{cases}
$$
For any particular loss function, the empirical risk that we minimize is then:  
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \varphi\left(y^{(i)}\theta^T x^{(i)}\right).
$$
Through this jacobian, we can penalize those $\theta$ for which
$y^{(i)}\theta^Tx^{(i)} < 0$ frequently in the training data(loss increases by 1 for each data), and record no loss for $y^{(i)}\theta^Tx^{(i)} > 0$ for each training examples.  
Choosing $\theta$ to minimize the average logistic loss will
yield a $\theta$ for which $y^{(i)}\theta^Tx^{(i)} > 0$ for most (or even all!) of the training examples.  

#### Types of Loss Functions
 - logisitc loss: $\varphi_{\text{logistic}}(z) = \log(1 + e^{-z})$  
 - hinge loss: $\varphi_{\text{hinge}}(z) = [1 - z]_+ = \max\{1 - z, 0\}$  
 - exponential loss: $\varphi_{\exp}(z) = e^{-z}$  

#### Probabilistic intrepretation
When hypothesis as,  
$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}},
$$
Using logistic loss function, loss is,  
$$
\varphi_{\text{logistic}}(yx^T\theta) = \log\left(1 + \exp(-yx^T\theta)\right)
$$
And Choose $\theta$ that minimizes $J(\theta)$ which is logistic regression risk is,  
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \varphi_{\text{logistic}}(y^{(i)}\theta^T x^{(i)}) = \frac{1}{m} \sum_{i=1}^{m} \log \left( 1 + \exp(-y^{(i)}\theta^T x^{(i)}) \right).
$$
The likelihood of the training data is,   
$$
L(\theta) = \prod_{i=1}^{m} p(Y = y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^{m} h_\theta(y^{(i)} x^{(i)})
$$  
So the log-likelihood is,  
$$
\ell(\theta) = \sum_{i=1}^{m} \log h_\theta(y^{(i)} x^{(i)}) = - \sum_{i=1}^{m} \log \left( 1 + e^{-y^{(i)} \theta^T x^{(i)}} \right) = -m J(\theta)
$$
That is, maximum likelihood in the logistic model $L(\theta)$ is the same as minimizing the average logistic loss $J(\theta)$, and we arrive at logistic regression again.

### Gradient Descent in Logistic Regression
Consider gradient-descent-based procedures for performing the minimization of logistic loss.  
With that in mind, the derivatives of the logistic loss is,  
$$
\begin{align*}
\frac{d}{dz} \varphi_{\text{logistic}}(z) &= \varphi_{\text{logistic}}'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{d}{dz} e^{-z} = -\frac{e^{-z}}{1 + e^{-z}} = -\frac{1}{1 + e^{z}} = -g(-z), \\
\end{align*}
$$
when $z = y \theta^T x$,  
$$
\begin{align*}
\frac{\partial}{\partial \theta_k} \varphi_{\text{logistic}}(y\theta^Tx) &= -g(-y\theta^Tx) \frac{\partial}{\partial \theta_k} (y\theta^Tx) = -g(-y\theta^Tx) yx_k.
\end{align*}
$$
Thus, a stochastic gradient procedure for minimization of $J(\theta)$  iteratively performs the following for every iteraion.  
 1. choose $i \in \{1, \cdots, m\}$ uniformly at random.
 2. perform gradient update rule:  
 $\theta^{(t+1)} = \theta^{(t)} - \alpha_t \cdot \nabla_{\theta} \varphi_{\text{logistic}}(y^{(i)} x^{(i)T} \theta^{(t)})$  
 $\theta^{(t+1)} = \theta^{(t)} + \alpha_t g(-y^{(i)} x^{(i)T} \theta^{(t)}) y^{(i)} x^{(i)} = \theta^{(t)} + \alpha_t h_{\theta(t)}(-y^{(i)} x^{(i)}) y^{(i)} x^{(i)}$  

This update is intuitive:
If our current hypothesis $h_{\theta(t)}$ assigns probability close to 1 for the incorrect label $y(i)$, then we try to reduce the loss by moving $\theta$ in the direction of $y(i)x(i)$.  
Conversely, if our current hypothesis $h_{\theta(t)}$ assigns probability close to 0 for the incorrect label $y(i)$, the update
essentially does nothing.

### Newton's Method
A root-finding algorithm which produces successively better approximations to the roots (or zeroes) of a real-valued function.  
It starts with a real-valued function $f$, its derivative $f′$, and an initial guess $x_0$ for a root of $f$.  
If $f$ satisfies certain assumptions and the initial guess is close, then
$$
x_1 = x_0 - \frac{f(x_0)}{f'(x_0)}
$$
$x_1$ is a better approximation of the root than $x_0$. The process is repeated as,  
$$
x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
$$
## Softmax Regression
The softmax function takes as input a vector $z$ of $K$ real numbers, and normalizes it into a probability distribution consisting of $K$ probabilities proportional to the exponentials of the input numbers.  
That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval $(0,1)$.  
Formally, the standard (unit) softmax function $\sigma$:  
$$
\sigma: \mathbb{R}^K \rightarrow (0, 1)^K, \text{ where } K \geq 1,
$$  
$$
\text{vector } \mathbf{z} = (z_1, \dots, z_K) \in \mathbb{R}^K
$$
$$
\sigma(\mathbf{z}) \in (0,1)^K \text{ with } \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}.
$$
The softmax applies the standard exponential function to each element 
$z_{i}$ of the input vector $z$ (consisting of $K$), and normalizes these values by dividing by the sum of all these exponentials.  
The normalization ensures that the sum of the components of the output vector $\sigma(z)$ is 1. 
### Multinomial Distribution
The multinomial distribution is a generalization of the binomial distribution.  
For $n$ independent trials each of which leads to a success for exactly one of $k$ categories, with each category having a given fixed success probability, the multinomial distribution gives the probability of any particular combination of numbers of successes for the various categories.  
Mathematically, if we have $k$ possible mutually exclusive outcomes, $x_i$ indicates the number of times outcome number $i$ is observed over the $n$ independent trials, with corresponding probabilities $p_1, \cdots, p_k$, Probability Mass Function(PMF) of multinomial distribution is:  
(어떤 시행애서 $k$ 가지의 값이 나타날 수 있고, 그 값들이 나타날 확률이 각각 $p_1, \cdots, p_k$ 라고 할 때 $n$번의 독립 시행에서 $i$ 번째의 값이 $x_i$ 번 나타날 확률)
$$
f(x_1, \dots, x_k; n, p_1, \dots, p_k) = \Pr(X_1 = x_1 \text{ and } \dots \text{ and } X_k = x_k) = \\ 
\begin{cases} 
\frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \times \cdots \times p_k^{x_k}, & \text{when } \sum_{i=1}^k x_i = n \\
0, & \text{otherwise}
\end{cases}
$$

#### Applying Multinomial Distribution
In our model, we have one trial($n=1$) and the number of times outcome number $i$ is one.($x_k=1$). So the modified distribution is:  
$$
\frac{1!}{1! \cdots 1!} p_1^{x_1} \times \cdots \times p_k^{x_k} = \\
p_1^{x_1} \times \cdots \times p_k^{x_k} = \\
\prod_{i}^{k} P_{i}^{y_{i}}
$$

### Multinomial Distribution as GLM
When $\phi_i \cdots \phi_k$ specifying the probability of each of the outcomes:
$$
\phi_i = p(y = i; \phi), \quad \text{and} \quad p(y = k; \phi) = 1 - \sum_{i=1}^{k-1} \phi_i
$$  
To express the multinomial as an exponential family distribution, define $T(y) \in \mathbb{R}^{k-1}$ as follow:  
$$
\begin{align*}
T(1) &= \begin{pmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad 
T(2) = \begin{pmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad  \dots, \quad
T(k-1) = \begin{pmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 1 \end{pmatrix}, \quad 
T(k) = \begin{pmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix},
\end{align*}
$$  
In this case, $T(y)$ !=  $y$ and $T(y)_i$ is $i$-th element of the vector $T(Y)$.  
Also, $\left(T(y)\right)_i = \mathbf{1}\{y = i\}$ and $\mathbb{E}\left[(T(y))_i\right] = P(y = i) = \phi_i$.  
$$
\begin{aligned}
p(y;\boldsymbol{\phi}) &= \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \cdots \phi_k^{1\{y=k\}} \\
&= \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \cdots \phi_i^{1\{y=i\}} \cdots \phi_k^{1-\sum_{i=1}^{k-1} 1\{y=i\}} \\
&= \phi_1^{(T(y))_1} \phi_2^{(T(y))_2} \cdots \phi_k^{1-\sum_{i=1}^{k-1} (T(y))_i} \\
&= \exp\left((T(y))_1 \log(\phi_1) + (T(y))_2 \log(\phi_2) + \cdots + \left(1 - \sum_{i=1}^{k-1} (T(y))_i\right) \log(\phi_k)\right) \\
&= \exp\left((T(y))_1 \log(\phi_1/\phi_k) + (T(y))_2 \log(\phi_2/\phi_k) + \cdots + (T(y))_{k-1} \log(\phi_{k-1}/\phi_k) + \log(\phi_k)\right) \\
&= b(y) \exp(\boldsymbol{\eta}^T T(y) - a(\boldsymbol{\eta}))
\end{aligned}
$$
and,  
$$
\begin{aligned}
\text{where} \quad \boldsymbol{\eta} &= 
\begin{bmatrix}
\log(\phi_1/\phi_k) \\
\log(\phi_2/\phi_k) \\
\vdots \\
\log(\phi_{k-1}/\phi_k)
\end{bmatrix}, \\
a(\boldsymbol{\eta}) &= -\log(\phi_k), \\
b(y) &= 1.
\end{aligned}
$$ 
From Above, we can find the link function is $\eta_i = \log\left(\frac{\phi_i}{\phi_k}\right)
$, and response function is $e^{\eta_i} = \frac{\phi_i}{\phi_k}$.  
Using response function, 
$$
e^{\eta_i} = \frac{\phi_i}{\phi_k} \\
\phi_k e^{\eta_i} = \phi_i \\
\phi_k \sum_{i=1}^{k} e^{\eta_i} = \sum_{i=1}^{k} \phi_i = 1 \\
\phi_k = \frac{1}{\sum_{i=1}^{k} e^{\eta_i}} \\
$$
Thus, 
$$
\phi_i = \frac{e^{\eta_i}}{\sum_{j=1}^{k} e^{\eta_j}}
$$  
The function mapping from the $\eta$ to $\phi$ is called softmax function.  
Using Assumption 3, we have $\eta_i = \theta_i^T x$, the model assumes that the conditional distribution of $y$ given $x$ is:  
$$
\begin{align*}
p(y = i \mid x; \theta) &= \phi_i \\
&= \frac{e^{\eta_i}}{\sum_{j=1}^{k} e^{\eta_j}} \\
&= \frac{e^{\theta_i^T x}}{\sum_{j=1}^{k} e^{\theta_j^T x}}
\end{align*}
$$  
When applying this model to classification problems where $y \in \{1, \dots, k\}$, the hypothesis $h_{\theta}(x)$ is:  
$$
\begin{align*}
h_{\theta}(x) &= \mathbb{E}[T(y) \mid x; \theta] \\
&= \mathbb{E} \left[ \begin{pmatrix} 
1\{y=1\} \\ 
1\{y=2\} \\ 
\vdots \\ 
1\{y=k-1\} 
\end{pmatrix} \mid x; \theta \right] \\
&= \begin{pmatrix} 
\phi_1 \\ 
\phi_2 \\ 
\vdots \\ 
\phi_{k-1} 
\end{pmatrix} \\
&= \begin{pmatrix} 
\frac{\exp(\theta_1^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)} \\ 
\frac{\exp(\theta_2^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)} \\ 
\vdots \\ 
\frac{\exp(\theta_{k-1}^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)} 
\end{pmatrix}.
\end{align*}
$$  
In other words, our hypothesis will output the estimated probability $p(y = i \mid x; \theta)$ for every value of $\{1, \cdots, k\}$.  
Similar to logistic regression, we can learn the parameter $\theta$ by optaining the maximum likelihood estimate of the parameters by maximizing log-likelihood function:  
$$
\ell(\theta) = \sum_{i=1}^{m} \log p(y^{(i)}|x^{(i)}; \theta) \\
= \sum_{i=1}^{m} \log \prod_{l=1}^{k} \left( \frac{e^{\theta_l^T x^{(i)}}}{\sum_{j=1}^{k} e^{\theta_j^T x^{(i)}}} \right)^{1\{y^{(i)}=l\}}
$$
Like logisstic regression, maximum likelihood estimate can be obtained by using method gradient ascent or Newton’s method.