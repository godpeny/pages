# Generative Learning Algorithms (GLA)
## Basics (Discriminative Models vs Generative Models)
A generative model for images might capture correlations like "things that look like boats are probably going to appear near things that look like water" and "eyes are unlikely to appear on foreheads." These are very complicated distributions.  
In contrast, a discriminative model might learn the difference between "sailboat" or "not sailboat" by just looking for a few tell-tale patterns. It could ignore many of the correlations that the generative model must get right.  
Discriminative models try to draw boundaries in the data space, while generative models try to model how data is placed throughout the space. For example, the following diagram shows discriminative and generative models of handwritten digits:
![alt text](images/blog9_dlm_vs_glm.png)  
The discriminative model tries to tell the difference between handwritten 0's and 1's by drawing a line in the data space. If it gets the line right, it can distinguish 0's from 1's without ever having to model exactly where the instances are placed in the data space on either side of the line.

In contrast, the generative model tries to produce convincing 1's and 0's by generating digits that fall close to their real counterparts in the data space. It has to model the distribution throughout the data space.

### Generative Learning Models (GLA)
 - Generative model is a model of the conditional probability of the observable $X$, given a target $y$, symbolically, $P(Y \mid X=x)$
 - Generative model is a statistical model of the joint probability distribution $P(X, Y)$ on a given observable variable $X$ and target variable $Y$. A generative model can be used to "generate" random instances (outcomes) of an observation $x$.
 - Generative model includes the distribution of the data itself, and tells you how likely a given example is. e.g., models that predict the next word in a sequence are typically generative models (usually much simpler than GANs) because they can assign a probability to a sequence of words.
 - For example, Naive Bayes classifier, GAN and Gaussian Discriminant Analysis(GDA).   
### Discriminative Learning Models (DLA)
 - Discriminative model is a model of the conditional probability of the target $Y$, given an observation $x$, symbolically, $P(Y \mid X=x)$.
 - Discriminative model is a model of the conditional probability $P(Y\mid X=x)$ of the target $Y$, given an observation $x$. It can be used to "discriminate" the value of the target variable $Y$, given an observation $x$.
 - Discriminative model ignores the question of whether a given instance is likely, and just tells you how likely a label is to apply to the instance.
 - For example, Logistic Regression and Decision Tree

### Bayes' Rule (Theorem)
$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}, \ \ \text{when } P(B) \neq 0
$$
Also,  
$$
P(X) = P(X \mid Y = 1)P(Y = 1) + P(X \mid Y = 0)P(Y = 0)
$$
#### Applying Bayes' Rule to Conditional Probability
$$
p(\theta \mid x, y) = \frac{p(x, y, \theta)}{p(x, y)} =
\frac{p(y \mid x, \theta) p(x, \theta)}{p(x, y)} = \frac{p(y \mid x, \theta) p(\theta \mid x) p(x)}{p(x, y)}
$$
check how conditional probability represented, $p(\theta \mid x, y) = \frac{p(x, y, \theta)}{p(x, y)} $.


## Multivariate Normal Distribution
A generalization of the one-dimensional (univariate) normal distribution to higher dimensions.  
GDA is parameterized by a mean vector $\mu \in \mathbb{R}^{n}$ and a covariance matrix $\sum \in \mathbb{R}^{n}$  where $\sum \geq 0$ is symmetric and positive semi-definite. Also written $\mathcal{N}(\mu, \Sigma)$.  
The density of GDA:  
$$
p(x; \mu, \Sigma) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right).
$$  
For a random variable $X$ which is $m$-dimensional $X \in \mathbb{R}^m$,  distributed $\mathcal{N}(\mu, \Sigma)$, the mean $\mu$:  
$$
\mathbb{E}[X] = \int x \, p(x; \mu, \Sigma) \, dx = \mu \\
\text{Cov}(X) = \sum
$$
The covariance of a vector-valued random variable $Z$ is defined as  
$$
\text{Cov}(Z) = \mathbb{E}[(Z - \mathbb{E}[Z])(Z - \mathbb{E}[Z])^T]
$$
Also same as,  
$$
\text{Cov}(Z) = \mathbb{E}[ZZ^T] - (\mathbb{E}[Z])(\mathbb{E}[Z])^T.
$$

## Gaussian Discriminant Analysis (GDA)
Gaussian Discriminant Analysis (GDA) is a supervised learning algorithm used for classification tasks in machine learning.  
GDA works by assuming that the data(input features $x$ are
continuous-valued random variables) in each class follows a Gaussian distribution, and then estimating the mean and covariance matrix for each class. It then uses Bayes’ theorem to compute the probability that a new data point belongs to each class, and chooses the class with the highest probability as the predicted class.  
The model is:  
$$
\begin{align*}
y &\sim \text{Bernoulli}(\phi) \\
x \mid y = 0 &\sim \mathcal{N}(\mu_0, \Sigma) \\
x \mid y = 1 &\sim \mathcal{N}(\mu_1, \Sigma)
\end{align*}
$$
The distribution of GDA model is:  
$$
\begin{align*}
p(y) &= \phi^y (1 - \phi)^{1 - y} \\
p(x \mid y = 0) &= \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0)\right) \\
p(x \mid y = 1) &= \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1)\right)
\end{align*}
$$
Note that there are two mean vectore $\mu_0$ and $\mu_1$.  
Log likelihood is:  
$$
\begin{align*}
\ell(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) \\
&= \log \prod_{i=1}^{m} p(x^{(i)} \mid y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi).
\end{align*}
$$
By maximizing $\ell$ with respect to the parameters($ \phi, \mu_0, \mu_1, \Sigma$), we find the maximum likelihood estimate of the parameters to be:  
$$
\begin{align*}
\phi &= \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}\{y^{(i)} = 1\} \\
\mu_0 &= \frac{\sum_{i=1}^{m} \mathbb{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^{m} \mathbb{1}\{y^{(i)} = 0\}} \\
\mu_1 &= \frac{\sum_{i=1}^{m} \mathbb{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^{m} \mathbb{1}\{y^{(i)} = 1\}} \\
\Sigma &= \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T.
\end{align*}
$$

### GDA vs Logistic Regression

## Naive Bayes

### Laplace Smoothing

### Multi-Variate Bernoulli Event Model vs Multinomial Event Model