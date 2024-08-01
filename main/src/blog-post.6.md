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
That is, the entry of the $i$ th row and the $j$ th column is 
$
\left( \mathbf{H}_f \right)_{i,j} = \frac{\partial^2 f}{\partial x_i \partial x_j}.
$

### Jacobian
The Jacobian matrix of a vector-valued function of several variables is the matrix of all its first-order partial derivatives.  
Suuppose $\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m
$ is a function such that each of its first-order partial derivatives exists on $\mathbb{R}^n$.  
This function takes a point $\mathbf{x} \in \mathbb{R}^n$ as input and produces the vector $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^m$ as output.  
Then the Jacobian matrix of $\mathbf{f}$ is defined to be an $m \times n$ matrix, denoted by $\mathbf{J}$, whose $(i,j)$ entry is $\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$, or explicitly  
$
\mathbf{J} = \left[ \frac{\partial \mathbf{f}}{\partial x_1} \ \cdots \ \frac{\partial \mathbf{f}}{\partial x_n} \right] = \left[ \begin{array}{c} \nabla^T f_1 \\ \vdots \\ \nabla^T f_m \end{array} \right] = \left[ \begin{array}{ccc} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{array} \right]
$  
where $\nabla^T f_1$ is the transpose (row vector) of the gradient of the $i$ th component.  
The Jacobian matrix represents the differential of $f$ at every point where $f$ is differentiable.  
즉, Jacobian(야코비언)은 어떤 다변수 벡터함수(vector-valued function of multiple variables)에 대한 일차 미분(first derivative)으로 볼 수 있습니다.

### Gradient vs Jacobian
- 공통점 : Gradient나 Jacobian이나 모두 함수에 대한 일차 미분(first derivative)을 나타낸다는 점에서는 동일합니다.  
- 차이점 : Gradient나는 다변수 스칼라 함수(scalar-valued function of multiple variables)에 대한 일차 미분인 반면 Jacobian은 다변수 벡터 함수(vector-valued function of multiple variables)에 대한 일차미분입니다. 즉, Gradient는 통상적인 일변수 함수의 일차미분을 다변수 함수로 확장한 것이고, Jacobian은 이를 다시 다변수 벡터함수로 확장한 것입니다.

#### Gradient
$
f(x,y) = 5x^2 + 3xy + 3y^3, \\
\frac{\partial f}{\partial x} = 10x + 3y, \ \ \frac{\partial f}{\partial y} = 3x + 9y^2 \\
\nabla f = \begin{bmatrix}
10x + 3y \\
3x + 9y^2
\end{bmatrix}
$

#### Jacobian
$
\mathbf{J}\mathbf{h}(f(x,y),g(x,y)) = \begin{bmatrix} f_x & f_y \\ g_x & g_y \end{bmatrix}, \\
f(x,y) : \begin{bmatrix} \sin(x) + y \\ x + \cos(y) \end{bmatrix} \\
\mathbf{J}(f) = \begin{bmatrix} \cos(x) & 1 \\ 1 & -\sin(y) \end{bmatrix}
$  
$\mathbf{J}(f)$ shows the jacobian of function $f$ from vector-valued function of multiple variables $\mathbf{h}$.  


## Differential vs Derivative vs Gradient
## Epsilon in Calculus
- Epsilon is a small positive number, often used in numerical computation to avoid division by zero or taking the logarithm of zero.

## Epsilon-Delta Definition of Limit