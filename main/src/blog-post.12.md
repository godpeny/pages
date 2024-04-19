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
Lagrange dual function $g(\lambda, \nu)$ is defined as below,
$$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) = \inf_{x \in D} \left( f_0(x) + \sum_{i=1}^{m} \lambda_i f_i(x) + \sum_{i=1}^{p} \nu_i h_i(x) \right). \\
p^* \text{ is optimal value from primal problem.}
$$

for any $\lambda \geq 0$  and any $\nu $ we have, $g(\lambda, \nu) \leq p^*$.  
for each pair $(\lambda, \nu)$ with $\lambda \geq 0$, the Lagrange dual function gives us a lower bound on the optimal value $ p⋆ $ of the optimization problem.  
thus have a lower bound that depends on some parameters $\lambda$, $\nu$.

#### lower value on optimal value
for feasible $\hat{x}$ which satisfy constraints,  
$$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) \leq L(\hat{x}, \lambda, \nu) \leq f_0(\hat{x}).
$$ 
this is because if Lagrangian is unbounded below in x(e.g. $\inf_{x \in D} L(x, \lambda, \nu)$),  
the dual function takes on the value $-\infty$.

#### Lagrange Dual Problem
$$
\begin{align*}
& \text{maximize} & & g(\lambda, \nu) \\
& \text{subject to} & & \lambda \geq 0.
\end{align*}
$$
- optimization problem to find the best lower bound that can be obtained from the Lagrange dual function.
- the Lagrange dual problem is a convex optimization problem, since the
objective to be maximized is concave and the constraint is convex.(This is the case
whether or not the primal problem is convex.)

#### Example 1 - Linear Programming
for the primal problem,
$$
\begin{align*}
& \text{minimize} & & c^T x \\
& \text{subject to} & & Ax = b, \\
& & & Gx \leq h.
\end{align*}
$$
using lagrange multiplier vector $u$ and $v$, make Lagrangian $L$,
$$
L(x, u, v) = c^T x + u^T (Ax - b) + v^T (Gx - h) \leq c^T x \\
\text{when vector} \quad v \geq 0 
$$ 
this is because $u^T (Ax - b)$ is always zero and $v^T (Gx - h)$ is always zero or negative. ($Ax = b, \ Gx \leq h$)
when lagrange optinal function is $g(\lambda, \nu) = \min_{x} L(x, u, v)$, 
$$
p^* \geq \min_{x \in C} L(x, u, v) \geq \min_{x} L(x, u, v) \\
\text{when C is feasible region that meets constraints, } \\
p^* \text{is optimal value that we are looking for.}
$$
this is because you can better minimize when constraints is not set. (=lower value on optimal value)
let's partial differentiate $g(\lambda)=\min_{x} L(x, u, v)$ w.r.t $x$,
$$
\frac{\partial L}{\partial x} = c^T + u^T A + v^T G = 0 \\
\therefore c = -A^T u - G^T v
$$

using this equation, Lagrangian $L$ is,
$$
L(x, u, v) = c^T x + u^T (Ax - b) + v^T (Gx - h) \\
\implies (-A^T u - G^T v) x + u^T (Ax - b) + v^T (Gx - h) \\
= -u^T Ax - v^T Gx + u^T Ax - u^T b + v^T Gx - v^T h \\
= -u^T b - v^T h \\
= g(u, v)
$$

so original primal problem is same as maximizing $g(u, v) =  L(x, u, v)$ since, $ p^* \geq L(x, u, v)$.

#### Example 2 - General Case
consider the primal problem,
$$
\begin{align*}
& \text{minimize} & & f(x) \\
& \text{subject to} & & h_i(x) \leq 0, & & i = 1, \ldots, m, \\
& & & l_j(x) = 0, & & j = 1, \ldots, r.
\end{align*}
$$
let's define Lagrangian $L$ and Lagrangian dual problem $g$ as below,
$$
L(x, u, v) = f(x) + \sum_{i=1}^{m} u_i h_i(x) + \sum_{i=1}^{r} v_i l_i(x) \\
g(u, v) = \min_{x} L(x, u, v) \\
\text{when vector} \quad u \geq 0
$$
since  $p^* \geq L(x, u, v)$, prime problem which is looking for only vector $x$ can be changed into dual problem as below which is looking for vector $v$, $u$.
$$
\begin{align*}
& \text{maximize} & & g(u, v) \\
& \text{subject to} & & u \geq 0.
\end{align*}
$$
you can also see that dual problem is maximizing problem while prime problem was minimizaing problem.

#### Weak Duality, Strong Duality and Duality Gap
The optimal value of the Lagrange dual problem, which we denote $d^⋆$, is, by definition,
the best lower bound on $p^⋆$ that can be obtained from the Lagrange dual
function. It can be described as below,
$$
d^*\leq p^*
$$
above inequality property is called weak duality.
$$
d^* = p^*
$$
above equality property is called strong duality.  
duality gap is basically the gap between the optimal value of the primal problem ($p^*$)
and the best (i.e., greatest) lower bound on it($d^*$) that can be obtained from the
Lagrange dual function. It can be described as below,
$$
p^* - d^*
$$


### Karush-Kuhn-Tucker(KKT) Conditions
### KKT Dual Complementarity Condition