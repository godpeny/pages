# Lagrange Duality

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

## Lagrange multiplier
$\lambda$, $\nu$ are called the dual variables or lagrange multiplier vectors.
  - $\lambda_i$ : lagrange multiplier associated with the $i$ th inequality constraint,  $f_i(x) \leq 0$
  - $\nu_i$ : lagrange multiplier associated with the $i$ th equality constraint, $h_i(x) = 0$

## Lagrange dual function
Lagrange dual function $g(\lambda, \nu)$ is defined as below,
$$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) = \inf_{x \in D} \left( f_0(x) + \sum_{i=1}^{m} \lambda_i f_i(x) + \sum_{i=1}^{p} \nu_i h_i(x) \right). \\
p^* \text{ is optimal value from primal problem.}
$$

for any $\lambda \geq 0$  and any $\nu $ we have, $g(\lambda, \nu) \leq p^*$.  
for each pair $(\lambda, \nu)$ with $\lambda \geq 0$, the Lagrange dual function gives us a lower bound on the optimal value $ p⋆ $ of the optimization problem.  
thus have a lower bound that depends on some parameters $\lambda$, $\nu$.

## lower value on optimal value
for feasible $\hat{x}$ which satisfy constraints,  
$$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) \leq L(\hat{x}, \lambda, \nu) \leq f_0(\hat{x}).
$$ 
this is because if Lagrangian is unbounded below in x(e.g. $\inf_{x \in D} L(x, \lambda, \nu)$),  
the dual function takes on the value $-\infty$.

## Lagrange Dual Problem
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

## Example 1 - Linear Programming
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

## Example 2 - General Case
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

## Weak Duality, Strong Duality and Duality Gap
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

## Constraint Qualifications for Strong Duality
there are various ways to guarantee 'strong duality"' in optimization problems, even when the problem isn't necessarily convex. and these conditions are called 'constraint qualifications'.  
for example, 
$$
\begin{align*}
& \text{minimize} & & f(x) \\
& \text{subject to} & & h_i(x) \leq 0, & & i = 1, \ldots, m, \\
& & & l_j(x) = 0, & & j = 1, \ldots, r. \\
& \text{when f(x) is convex}
\end{align*}
$$
above form is one example of constraint. another simple constraint qualification is 'Slater’s condition'.

## Slater’s condition
a sufficient condition for strong duality to hold for a convex optimization problem.  
There exists an $x \in \text{relint}\, D$ such that,
$$
f_i(x) < 0, \quad i = 1, \ldots, m, \quad Ax = b.
$$

## Karush-Kuhn-Tucker(KKT) Conditions
### KKT conditions for nonconvex problems
let $x^⋆$ and $(\lambda^*, v^*)$ be any primal and dual optimal points with zero duality gap (strong duality).  
since $x^*$ minimizes $L(x^*, \lambda^*, v^*)$ over x, it follows that its gradient must vanish at $x^*$. In other words,
$$
\nabla f_0(x^*) + \sum_{i=1}^{m} \lambda_i^* \nabla f_i(x^*) + \sum_{i=1}^{p} \nu_i^* \nabla h_i(x^*) = 0.
$$
then any pair of primal and dual optimal point must satisfy KKT conditons below.
$$
\begin{align*}
&\nabla f_0(x^*) + \sum_{i=1}^{m} \lambda_i^* \nabla f_i(x^*) + \sum_{i=1}^{p} \nu_i^* \nabla h_i(x^*) = 0, \\
&\lambda_i^* f_i(x^*) = 0, \quad i = 1, \ldots, m, \\ 
&f_i(x^*) \leq 0, \quad i = 1, \ldots, m, \\
&h_i(x^*) = 0, \quad i = 1, \ldots, p, \\
&\lambda_i^* \geq 0, \quad i = 1, \ldots, m.
\end{align*}
$$
to sum up, for any optimization problem with differentiable objective and
constraint functions for which strong duality obtains, any pair of primal and dual
optimal points must satisfy the KKT conditions.

### KKT conditions for convex problems
When the primal problem is convex, the KKT conditions are also sufficient for the
points to be primal and dual optimal.  
In other words, if $f_i$ are convex and $h_i$ are
affine, and $\hat{x}$, $\hat{\lambda}$ and $\hat{v}$ are any points that satisfy the KKT conditions,
$$
\begin{align*}
&\nabla f_0(\hat{x}) + \sum_{i=1}^{m} \hat{\lambda}_i \nabla f_i(\hat{x}) + \sum_{i=1}^{p} \hat{\nu}_i \nabla h_i(\hat{x}) = 0, \\
&\hat{\lambda}_i f_i(\hat{x}) = 0, \quad i = 1, \ldots, m, \\ 
&f_i(\hat{x}) \leq 0, \quad i = 1, \ldots, m, \\
&h_i(\hat{x}) = 0, \quad i = 1, \ldots, p, \\
&\hat{\lambda}_i \geq 0, \quad i = 1, \ldots, m.
\end{align*}
$$

### Necessity
Necessity will show that ,
If $x^*$ and $u^*, v^*$ are primal and dual solutions, with zero duality gap,  
$\implies$ $x^*, u^*, v^*$ satisfy the KKT conditions.

remind that, 
$$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) = \min_{x} L(x, \lambda,\nu)
$$
and primal solution is $x^*$ and dual solution are $\lambda^*$, $\nu^*$.  
under strong duality,
$$
f_0(x^*) = g(\lambda^*, \nu^*)
$$
since $\inf_{x \in D} L(x, \lambda^*, \nu^*)$ is lower than $L(x, \lambda^*, \nu^*)$ with any other x, it is also lower than $L$ with primal solution $x^*$. 
$$
\inf_{x \in D} L(x, \lambda^*, \nu^*) \leq L(x^*, \lambda^*, \nu^*), \\ 
$$
therefore, 
$$
f_0(x^*) = g(\lambda^*, \nu^*) = \inf_{x \in D} L(x, \lambda^*, \nu^*) \leq L(x^*, \lambda^*, \nu^*) =  f_0(x^*) + \sum_{i=1}^{m} \lambda_i f_i(x^*) + \sum_{i=1}^{p} \nu_i h_i(x^*)

$$
since $x^*$ is primal optimal, it satisfies primal constraints $f_i(x^*) \leq 0$ and $h_i(x^*) = 0$.  
also, considering inequality of primal problem $f_0(x^*)$, Lagrangian $L$ and Lagrange dual problem $g(\lambda^*, \nu^*)$, ('lower value on optimal value')
$$
f_0(x^*) = g(\lambda^*, \nu^*) = \inf_{x \in D} L(x, \lambda^*, \nu^*) \leq L(x^*, \lambda^*, \nu^*) = f_0(x^*) + \sum_{i=1}^{m} \lambda_i f_i(x^*) + \sum_{i=1}^{p} \nu_i h_i(x^*) \leq f_0(x^*)
$$
equality('=') works for above. this shows that 
$$
\text{1.} \inf_{x \in D} L(x, \lambda^*, \nu^*) = L(x^*, \lambda^*, \nu^*) \\
\text{2.} \sum_{i=1}^{p} \lambda_i f_i(x^*) =0
$$ 
'1.' shows that $L(x, \lambda^*, \nu^*)$ has the minimum when $x=x^*$ and it implies that its gradient must vanish at $x^*$, 
$$
\nabla f_0(x^*) + \sum_{i=1}^{m} \lambda_i^* \nabla f_i(x^*) + \sum_{i=1}^{p} \nu_i^* \nabla h_i(x^*) = 0, \\
$$
this shows that it satisfies KKT's first condition which is called 'stationarity'.  
'2.' satisfies KKT's last condition which is called 'complementary slackness' (i.e. 'KKT dual complementarity condition')

since primal solution is $x^*$ and dual solution are $\lambda^*$, $\nu^*$, it satisfies all other conditions of KKT. (primal condition : line3,4)

### Sufficiency
if $x^*, \lambda^*, \nu^*$ satisfies KKT conditions $\implies$ $x^*, \lambda^*, \nu^*$ are primal and dual solution with string duality.

since first KKT condition states that its gradient with respect to $x$ vanishes at $x=x^*$, so it follows that $x^*$ minimizes $L(x, \lambda^*, \nu^*)$ which means,
$$
\inf_{x \in D} L(x, \lambda^*, \nu^*) = L(x^*, \lambda^*, \nu^*)
$$
and it implies that,
$$
f_0(x^*) = g(\lambda^*, \nu^*) = \inf_{x \in D} L(x, \lambda^*, \nu^*)
$$
using the condition $\sum_{i=1}^{p} \lambda_i f_i(x)=0$ and $h_i(x^*) = 0$, 
$$
g(\lambda^*, \nu^*) = f_0(x^*) + \sum_{i=1}^{m} \lambda_i f_i(x^*) + \sum_{i=1}^{p} \nu_i h_i(x^*) = f_0(x^*) \\ 
\therefore g(\lambda^*, \nu^*) = f_0(x^*)
$$
this implies the strong duality. in other words, it shows that $x^*, \lambda^*, \nu^*$ are optimal solutions.

### Conclusion
If $x^*$ and $u^*, v^*$ are primal and dual solutions, with zero duality gap $\Leftrightarrow$
$x^*, \lambda^*, \nu^*$ satisfies KKT conditions.

## Reference
- https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
- https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/kkt.pdf
- https://ratsgo.github.io/convex%20optimization/2018/01/25/duality/
- https://ratsgo.github.io/convex%20optimization/2018/01/26/KKT/
- https://lee-jaejoon.github.io/optimization-lagrange-kkt/