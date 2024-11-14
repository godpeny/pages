# Support Vector Machines (SVM)

## Unit Vector
In mathematics, a unit vector in a normed vector space is a vector (often a spatial vector) of length 1.
$$
\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|}
$$ 
where $\|\mathbf{u}\|$ is the norm (or length) of $u$.  
The term normalized vector is sometimes used as a synonym for unit vector.  
A unit vector is often used to represent direction.

## Support Vector
Support Vectors are datapoints that are closest to the hyperplane.

## Margin
Margin is defined as the gap between two lines on the closet data points of different classes. It can be calculated as the perpendicular distance from the line to the support vectors. 

## Hyperplane
p차원 공간에서, 초평면은 p-1 차원인 평평한 affine 부분 공간 입니다. 예를 들어, 2차원 공간에서 초평면은 평평한 1차원 부분공간 입니다. 즉, 선 입니다. 3차원에서 초평면은 평평한 2차원 부분공간이며 이는 평면입니다.  
In geometry, a hyperplane is a generalization of a 2-D plane in 3-D space to mathematical spaces of arbitrary dimension.  
Like a plane in space, a hyperplane is a flat hypersurface, a subspace whose dimension is one less than that of the ambient space. Two lower-dimensional examples of hyperplanes are one-dimensional lines in a plane and zero-dimensional points on a line.

Most commonly, the ambient space is n-dimensional Euclidean space, in which case the hyperplanes are the (n − 1)-dimensional "flats", each of which separates the space into two half spaces.

In ML, hyperplanes are essentially a boundary which classifies the data set (classifies Spam email from the ham ones). It could be lines, 2D planes, or even n-dimensional planes that are beyond our imagination.

### Kinds of Hyperplanes in SVM
#### Optimal Hyperplane
This is the main decision boundary that separates the two classes of data points. The goal of SVM is to find the hyperplane that maximizes the margin between two classes.
#### Positive Hyperplane:
The positive hyperplane is the boundary on one side of the margin where the support vectors of the positive class (let’s say class +1) lie.  
Data points from the positive class lie either on or beyond this boundary.
#### Negative Hyperplane:
The negative hyperplane is the boundary on the opposite side of the margin where the support vectors of the negative class (let’s say class -1) lie.  
Data points from the negative class lie either on or beyond this boundary.

### Relationship of Support Vector, Margin and Hyperplane
#### 1
![alt text](images/blog12_supportvector_margin_hyperplane.png)

#### 2
![alt text](images/blog12_supportvector_margin_hyperplane_2.png)


Where optimal hyperplane is,
$$
w^T x + b = 0 
$$
$w$ is weight vector, $b$ is bias, $x$ is feature vector.

For the positive hyperplane,
$$
w^T x + b = +1
$$
For the negative hyperplane.
$$
w^T x + b = -1
$$


## Functional Margin and Geometric Margin
The functional margin is the distance between the hyperplane (decision boundary) that separates the classes and the nearest training examples and it does not depend on the magnitude.
The functional margin is a measure of the generalization ability of the classifier. A larger functional margin indicates confident and a correct prediction.
$$
\hat{\gamma}^{(i)} = y^{(i)} (w^T x + b).
$$
The geometric margin is just a scaled version of the functional margin.
$$
\gamma^{(i)} = \frac{\hat{\gamma}^{(i)}}{\| \mathbf{w} \|}.
$$
![alt text](images/blog12_functional_and_geometric_margin.png)
When A is $x^{(i)}$ and $\frac{w}{\|w\|}$ is a unit-length vector pointing in the same direction as $w$, B is $x^{(i)} - \gamma^{(i)} \frac{w}{\|w\|}$.  
Since point B lies on the hyperplane which is $w^T x + b$,
$$
w^T \left( x^{(i)} - \gamma^{(i)} \frac{w}{\|w\|} \right) + b = 0.
$$
Solving for $\gamma^{(i)}$, 
$$
\quad \gamma^{(i)} = \frac{w^T x^{(i)} + b}{\|w\|} = \left( \frac{w}{\|w\|} \right)^T x^{(i)} + \frac{b}{\|w\|}.
$$
Remind that,
$$
\hat{\gamma}^{(i)} = y^{(i)} (w^T x + b).
$$
Therefore we can find out that geometric margin is scaled version of functonal margin.
$$
\gamma^{(i)} = y^{(i)} \left( \left( \frac{w}{\|w\|} \right)^T x^{(i)} + \frac{b}{\|w\|} \right) = \frac{\hat{\gamma}^{(i)}}{\| \mathbf{w} \|}
$$.
In other words, the geometric margin is invariant to the rescaling of the parameter, which is the only difference between geometric margin and functional margin.

## Optimal Margin Classifier
A margin classifier is a type of classification model which is able to give an associated distance from the decision boundary for each data sample.  
For instance, if a linear classifier is used, the distance (typically Euclidean, though others may be used) of a sample from the separating hyperplane is the margin of that sample.  
The Optimal margin classifier is the classifier that maximizes the margin between two classes. 


### Optimization Problem  

$$
\min_x \quad f_0(x) \\
\text{subject to :} \quad f_i(x) \leq b_i, \quad i = 1, \ldots, m. 
$$


making the best possible choice of a vector from a set of candidate choices.
- The variable x represents the choice made.
- the constraints fi(x) ≤ bi represent firm requirements or specifications that limit the possible choices.
- the objective value f0(x) represents the cost of choosing x. (or value of choosing x)

#### Convex Optimization Problem
$$
f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y) \\

\text{for all} \ \ x, y \in \mathbb{R}^n \quad \text{and all} \ \ \alpha, \beta \in \mathbb{R}, \quad \alpha + \beta = 1, \ \ \alpha \geq 0, \beta \geq 0 
$$

A convex optimization problem is optimization problem in which the objective and
constraint functions are convex, which means they satisfy the inequality above.

$$
\min_x \quad f(x) \\ 
\begin{align*}
\text{Subject to :} \quad f_i(x) \leq 0, \quad i = 1, \ldots, m \\
h_j(x) = 0, \quad j = 1, \ldots, p \\
\end{align*} \\


\begin{align}
f: \mathbb{R}^n \rightarrow \mathbb{R} \ \ \  \text{is the objective function to be minimized over the n-variable vector x.} \\ 
f_i(x) \leq 0 \quad \text{are called inequality constraints.} \\
h_j(x) = 0  \quad \text{are called equality constraints.} \\
m \geq 0 \quad \text{and} \quad p \geq 0 \\ 
\end{align}
$$
- The inequalities fi(x) ≤ 0 are called inequality constraints, and the
corresponding functions fi : Rn → R are called the inequality constraint functions.
- The equations hi(x) = 0 are called the equality constraints, and the functions
hj : Rn → R are the equality constraint functions.
- If there are no constraints (i.e.,m = p = 0) we say the problem is unconstrained.

### Primal Optimization Problem vs Dual Optimization Problem
duality is the principle that optimization problems may be viewed from either of two perspectives, the primal problem or the dual problem. If the primal is a minimization problem then the dual is a maximization problem (and vice versa). so lower bound of primal problem is upper bound of dual problem.

 - Primal Optimization Problem

$$
\min_x \quad \mathbf{c}^T \mathbf{x} \\

\begin{align*}
\text{subject to} \quad Ax=b, \\
Gx \leq h
\end{align*}
$$

 - Dual Optimization Problem

$$
\text{from Primal Optimzation Problem, applying vector } u \ \text{and } v, \quad v \geq 0 \\

u^T Ax = u^T b \\
u^T Gx \leq u^T h \\
\text{summing up above two equations} \\ 
u^T Ax + u^T Gx \leq u^T b + u^T h \\
(u^T A + u^T G)x \leq u^T b + u^T h \\
(A^T u + G^T u)^T x \leq u^T b + u^T h \\
(-A^T u - G^T u)^T x \geq -u^T b - u^T h \\
\therefore c^T x \geq -u^T b - u^T h \\

\text{Primal Optimization Problem turns into Dual Optimization Problem, } \\
\max_{u,v} \quad -b^T u - h^T v \\
\text{subject to} \quad -A^T u - G^T v = c \\
v \geq 0
$$
primal problem에서는 주어진 식을 만족하는 벡터 𝑥 를 찾는 것이었으나 dual problem에서는 벡터 𝑢,𝑣 를 찾는 문제로 바뀌었다.

## Support Vector
the points that are closest to the hyperplane.
- coefficients of support vectors($\alpha$) are the only ones that are non zero.
this is because margin of the support vectors is 1. which can be derived from the formulars from "Optimal Margin Classifier".

## Kernels
### Terminology
 - (input) attributes : original input value of problem. ($=x$)  
 - (input) features : new set of quantities. ($=\phi(x)$, e.g. [$x, x^2, x^3 ...$])

### Basics
#### kernel function
A function that takes as its inputs vectors in the original space and returns the dot product of the vectors in the feature space.  
loosely speaking, kernel function is dot product of the transformed vectors by considering that each coordinate of the transformed vector ϕ(x) is just some function of the coordinates in the corresponding lower dimensional vector x.  
For example,   
$$
\phi: X \rightarrow \mathbb{R}^N, k(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle
$$ 

$k(\mathbf{x}, \mathbf{z})$ is kernel function.   
It is an algorithms for pattern analysis, whose best known member is the support-vector machine (SVM). These methods involve using linear classifiers to solve nonlinear problems

#### kernel trick
Enable kernel functions to operate in a high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space. This operation is often computationally cheaper than the explicit computation of the coordinates.  
In other words, $K(x, y)$ may be very inexpensive to cacluate, even $\phi(x)$ and $\phi(z)$ may be very expensive to calculate. Thus, we can get SVMs to learn the high dimensional feature space given by $\phi$ but without ever having to explicitly find or represent vecotr $\phi(x)$.

#### Types of Kernels
 - polynomial kernel  
 $$
 K(x, y) = \langle \phi(x), \phi(y) \rangle = (x^\top y + c)^d
 $$
 - Gaussian kernel 

#### example of using polynomial kernel with kernel trick

## L1 Regularization

## Applying Kernel and Regularization to SVM

## Sequential Minimal Optimization (SMO)
Sequential minimal optimization (SMO) is an algorithm for solving the quadratic programming (QP) problem that arises during the training of support-vector machines (SVM). It was invented by John Platt.

### Basics
let's solve optimation problem from svm,
$$
\begin{align*}
\max_{\alpha} \quad & W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j K(x^{(i)}, x^{(j)}) \\
\text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1, \ldots, m, \\
& \sum_{i=1}^m \alpha_i y^{(i)} = 0,
\end{align*}
$$
repeat till convergence
1. Select some pair $a_i$ and $a_j$ to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum).  
2. Reoptimize W(α) with respect to αi and αj, while holding all the other $a_k$’s (k= i,j) fixed.
