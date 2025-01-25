# EM Algorithm
## K-means Clustering
K-means clustering is a method that aims to partition $n$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.

Given a training set $\{ x^{(1)}, \ldots, x^{(m)} \}$ and no labels $y^{(i}$, 

1. Initialize cluster centroids $\mu_1, \mu_2, \ldots, \mu_k \in \mathbb{R}^n$ randomly.

2. Repeat until convergence.  

For every $i$,
$$
c^{(i)} := \arg\min_j \|x^{(i)} - \mu_j\|^2.
$$
For each $j$,
$$
\mu_j := \frac{\sum_{i=1}^m 1\{c^{(i)} = j\} x^{(i)}}{\sum_{i=1}^m 1\{c^{(i)} = j\}}.
$$
Where , $k$ (a parameter of the algorithm) is the number of clusters we want to find, and the cluster centroids $\mu_j$ represent our current guesses for the positions of the centers of the clusters.

Mathmatically, partition the $n$ observations into $k (≤ n)$ sets $S = \{S_1, S_2, \cdots, S_k\}$ so as to minimize the within-cluster sum of squares (WCSS) (i.e. variance). 
$$
\arg\min_{\mathcal{S}} \sum_{i=1}^k \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \mu_i\|^2 = \arg\min_{\mathcal{S}} \sum_{i=1}^k |S_i| \operatorname{Var}(S_i)
$$
Where mean(cluster centroid) of points in $S_i$ is, 
$$
\mu_i = \frac{1}{|S_i|} \sum_{\mathbf{x} \in S_i} \mathbf{x},
$$

## Density Estimation
Density Estimation is the construction of an estimate of an unobservable underlying probability density function, based on observed data. A very natural use of density estimates is in the informal investigation of the properties of a given set of data.  
Therefore Density estimation is also frequently used in anomaly detection or novelty detection. If an observation lies in a very low-density region, it is likely to be an anomaly or a novelty.  
We can also assume that the observed data points of Density Estimation are distributed from multiple mixture of Gaussian distributions.

### Problem of Density Estimation
However the problem of Density Estimation is that you can only see the data came from set of Gaussains, but you don't know which example came from which Gaussian.  
Therefore Expectation-Maximization algorithm will allow us to fit the model despite not knowing which Gaussian each example that came from.

## Mixture of Gaussians

### Multivariate Gaussian, Mixture of Gaussian and Gaussian Discriminant Analysis(GDA)
#### GDA vs Mixture of Gaussian
 - In GDA, we have labled examples $(x_i, y_i)$ when $y_i$ is obseved. In Mixture of Gaussian, $y_i$ is replaced with $z_i$ which is latent(=hidden) random variable that we can not observer in the training set.
 - Mixture of Gaussian set $z(y)$ to one of $k$ values instead of two in GDA. In GDA, $y(z)$ takes on one of two values. 
 - Mixture of Gaussian use $\Sigma_{j}$ instead of $\Sigma$ in GDA, which means each Gaussian model in Mixture of Gaussian uses its own covariance matrix.

#### Multivariate Gaussian vs Mixture of Gaussian
While $k$ dimensional Multivariate Gaussian is, 
$$
\mathbf{X} = \begin{pmatrix} X_1, \dots, X_k \end{pmatrix} \sim \text{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),
$$
Where $\mu = \{\mu_1, \mu_2, \cdots, \mu_k \}$ is the mean vector, and $\Sigma$ is the positive definite $k \times k$ covariance matrix.

However, we can write out a $k$ component mixture of (1 dimensional) Gaussians as, 
$$
\mathbf{X} \sim \sum_{i=1}^{k} \pi_i \, \text{N}(\mu_i, \Sigma_i),
$$
where $\pi_i$ is the mixing proportion of the $i$-th component and $(\mu_i, \Sigma_i)$ are the parameters of the $i$-th component.  
As above, the Multivariate Gaussian is defined as a $k$ dimensional random vector, and the Mixture of Gaussians is defined as a random variable (which you can call a $1$ dimensional random vector).  

In conclusion, 
 - A Multivariate Gaussian Distribution describes a single, unified probability density over a vector of random variables, capturing the relationships (covariances) between them.  
 - A Mixture of Gaussians is a probabilistic model that represents a weighted sum of multiple Gaussian distributions. It is designed to model data that comes from several distinct subpopulations, each of which follows its own Gaussian distribution.

## Expectation–Maximization Algorithm (EM Algorithm)
An expectation–maximization (EM) algorithm is an iterative method to find (local) maximum likelihood estimates of parameters in statistical models, where the model depends on unobserved(latent) variables.  
The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step.   
These parameter-estimates are then used to determine the distribution of the latent variables in the next E step.  
It can be used, for example, to estimate a mixture of gaussians.

### Jensen's Inequality
Jensen's inequality generalizes the statement that the secant line(a line that intersects a curve at a minimum of two distinct points) of a convex function lies above the graph of the function. In the context of probability theory, it is generally stated in the following form: if $X$ is a random variable and $\mathbb{E}$ is a convex function, then
$$
\varphi(\mathbb{E}[X]) \leq \mathbb{E}[\varphi(X)]
$$
The difference between the two sides of the inequality is called the Jensen gap.
$$
\varphi(\mathbb{E}[X]) - \mathbb{E}[\varphi(X)]
$$

### Applying EM Algorithm to Mixture of Gaussians


## Factor Analysis