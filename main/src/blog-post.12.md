# Support Vector Machines (SVM)

## Margin
 - hyperplane

## Functional Margin and Geometric Margin

## Optimal Margin Classifier
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

### Lagrange Duality

$$
\begin{align*}
& \text{minimize} & & f_0(x) \\
& \text{subject to} & & f_i(x) \leq 0, & & i = 1, \ldots, m, \\
& & & h_i(x) = 0, & & i = 1, \ldots, p,
\end{align*}
$$
considering above optimization problem, 
$$
L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^{m} \lambda_i f_i(x) + \sum_{i=1}^{p} \nu_i h_i(x), \\

\text{when, Lagrangian } L : \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \rightarrow \mathbb{R}
$$

the basic idea in Lagrangian duality is to take the constraints into
account by augmenting the objective function with a weighted sum of the constraint
functions.

#### Lagrange multiplier
$\lambda$, $\nu$ are called the dual variables or lagrange multiplier vectors.
  - $\lambda_i$ : lagrange multiplier associated with the $i$ th inequality constraint,  $f_i(x) \leq 0$
  - $\nu_i$ : lagrange multiplier associated with the $i$ th equality constraint, $h_i(x) = 0$

#### Lagrange dual function
$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) = \inf_{x \in D} \left( f_0(x) + \sum_{i=1}^{m} \lambda_i f_i(x) + \sum_{i=1}^{p} \nu_i h_i(x) \right).
$

for any $\lambda \geq 0$  and any $\nu $ we have, $g(\lambda, \nu) \leq p^*$.


for each pair $(\lambda, \nu)$ with $\lambda \geq 0$, the Lagrange dual function gives us a lower bound on the optimal value $ p⋆ $ of the optimization problem.  
thus have a lower bound that depends on some parameters $\lambda$, $\nu$.



#### lower value on optimal value
for feasible $\hat{x}$ which satisfy constraints,  
$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) \leq L(\hat{x}, \lambda, \nu) \leq f_0(\hat{x}).
$  
this is because if Lagrangian is unbounded below in x(e.g. $\inf_{x \in D} L(x, \lambda, \nu)$),  
the dual function takes on the value $-\infty$.


### Karush-Kuhn-Tucker(KKT) Conditions
### KKT Dual Complementarity Condition