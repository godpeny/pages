# Calculus
## Chain Rule
## Jacobian vs Gradients vs Hessian vs Laplacian
https://darkpgmr.tistory.com/132

### Gradient
The gradient of a scalar-valued differentiable function $f$ of several variables is the vector field (or vector-valued function) $\nabla f$ whose value at a point $p$ gives the direction and the rate of fastest increase.  
When a coordinate system is used in which the basis vectors are not functions of position, the gradient is given by the vector whose components are the partial derivatives of $f$ at $p$.  
That is, for $f : \mathbb{R}^n \to \mathbb{R}$, its gradient $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$ is defined at the point $p = (x_1, \ldots, x_n)$ n n-dimensional space as the vector.  
$
[
\nabla f(p) = 
\frac{\partial f}{\partial x_1}(p) \\
\vdots \\
\frac{\partial f}{\partial x_n}(p)
]$

### Hessian
Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function, or scalar field.
Suppose $f : \mathbb{R}^n \to \mathbb{R}$ is a function taking as input a vector $\mathbf{x} \in \mathbb{R}^n$ and outputting a scalar $f(\mathbf{x}) \in \mathbb{R}$.  
If all second-order partial derivatives of $f$ exists, then the Hessian matrix $\mathbf{H}$ of $f$ is a square $n \times n$ matrix.  
$
\mathbf{H}_f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.
$


## Differential vs Derivative vs Gradient
## Epsilon in Calculus
- Epsilon is a small positive number, often used in numerical computation to avoid division by zero or taking the logarithm of zero.

## Epsilon-Delta Definition of Limit