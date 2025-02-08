# Principal Component Analysis (PCA)
## Preliminaries
### Eigenvector and Eigenvalue
An eigenvector is a vector that has its direction unchanged by a given linear transformation.  
Consider an $n{\times }n$ matrix A and a nonzero vector $v$ of length $n$. 
If multiplying $A$ with $v$ (denoted by $A v$) simply scales $v$ by a factor of $\lambda$, where $\lambda$ is a scalar, then $v# is called an eigenvector of $A$, and $\lambda$ is the corresponding eigenvalue. This relationship can be expressed as,
$$
A v =\lambda v
$$
Above formula can be stated equivalently as $\left(A - \lambda I \right)v = 0$
where $I$ is the $n \times n$ identity matrix and $0$ is the zero vector.
### Method of Lagrangie Multiplier
In mathematical optimization, the method of Lagrange multipliers is a strategy for finding the local maxima and minima of a function subject to equation constraints.  
(i.e., subject to the condition that one or more equations have to be satisfied exactly by the chosen values of the variables)

The basic idea is to convert a constrained problem into a form such that the derivative test of an unconstrained problem can still be applied.  
(In calculus, a derivative testuses the derivatives of a function to locate the critical points of a function and determine whether each point is a local maximum, a local minimum, or a saddle point) 

The relationship between the gradient of the function and gradients of the constraints rather naturally leads to a reformulation of the original problem, known as the Lagrangian function or Lagrangian.

#### Details
$$
\mathcal{L}(x, \lambda) \equiv f(x) + \langle \lambda, g(x) \rangle \equiv f(x) + \lambda \cdot g(x)

$$
Where $\langle \cdot, \cdot \rangle$ denotes dot product.  

The Method of Lagrangie Multiplier can be summarized as in order to find the maximum or minimum of a function $f$ subject to the equality constraint $g(x)=0$, find the stationary points(points on the graph of the differentiable function where the function's derivative is zero) of ${\mathcal {L}}$ considered as a function of $x$ and the Lagrange multiplier $\lambda$.  
This means that all partial derivatives should be zero, including the partial derivative with respect to $\lambda$.

$$
\frac{\partial f(x)}{\partial x} + \lambda \cdot \frac{\partial g(x)}{\partial x} = 0 \quad \text{and} \quad g(x) = 0
$$

#### Example
Suppose we wish to maximize, 
$$f(x,y)=x+y$$
Subject to the below constraint 
$$x^{2}+y^{2}=1$$
For the method of Lagrange multipliers, the constraint is, 
$$g(x,y)=x^{2}+y^{2}-1=0$$
Hence the Lagrangian function can be derived as below.
$$
\displaystyle {\begin{aligned}{\mathcal {L}}(x,y,\lambda )&=f(x,y)+\lambda \cdot g(x,y)\\[4pt]&=x+y+\lambda (x^{2}+y^{2}-1)\ ,\end{aligned}}
$$
This is a function that is equivalent to $f(x,y)$ when $g(x,y)=0$

Now we can calculate the gradient,
$$
\displaystyle {\begin{aligned}\nabla _{x,y,\lambda }{\mathcal {L}}(x,y,\lambda )&=\left({\frac {\partial {\mathcal {L}}}{\partial x}},{\frac {\partial {\mathcal {L}}}{\partial y}},{\frac {\partial {\mathcal {L}}}{\partial \lambda }}\right)\\[4pt]&=\left(1+2\lambda x,1+2\lambda y,x^{2}+y^{2}-1\right)\ \color {gray}{,}\end{aligned}}
$$
By setting to zero, we get,
$$
\displaystyle \nabla _{x,y,\lambda }{\mathcal {L}}(x,y,\lambda )=0\quad \Leftrightarrow \quad {\begin{cases}1+2\lambda x=0\\1+2\lambda y=0\\x^{2}+y^{2}-1=0\end{cases}}
$$

The first two equations yield,
$$
\displaystyle x=y=-{\frac {1}{2\lambda }},\qquad \lambda \neq 0
$$
By substituting into the last equation,
$$
\displaystyle {\frac {1}{4\lambda ^{2}}}+{\frac {1}{4\lambda ^{2}}}-1=0
$$

So $\displaystyle \lambda =\pm {\frac {1}{\sqrt {2\ }}}$ which implies that the stationary points of ${\mathcal {L}}$ are as below.
$$
\displaystyle \left({\tfrac {\sqrt {2\ }}{2}},{\tfrac {\sqrt {2\ }}{2}},-{\tfrac {1}{\sqrt {2\ }}}\right),\qquad \left(-{\tfrac {\sqrt {2\ }}{2}},-{\tfrac {\sqrt {2\ }}{2}},{\tfrac {1}{\sqrt {2\ }}}\right)
$$

Evaluating the objective function $f$ at these points yields
$$
\displaystyle f\left({\tfrac {\sqrt {2\ }}{2}},{\tfrac {\sqrt {2\ }}{2}}\right)={\sqrt {2\ }}\ ,\qquad f\left(-{\tfrac {\sqrt {2\ }}{2}},-{\tfrac {\sqrt {2\ }}{2}}\right)=-{\sqrt {2\ }}
$$
Thus, the constrained maximum is $\displaystyle \ {\sqrt {2\ }}$ and the constrained minimum is $\displaystyle -{\sqrt {2}}$.

## Basics
Principal component analysis is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize, and thus make analyzing data points much easier and faster for machine learning algorithms without extraneous variables to process.

In conclusionthe idea of PCA is simple: reduce the number of variables of a data set, while preserving as much information as possible.

## Pre-Processing

## Construction of PCA
