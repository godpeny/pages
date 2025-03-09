# Calculus
## Chain Rule
The chain rule is a formula that expresses the derivative of the composition of two differentiable functions $f$ and $g$ in terms of the derivatives of $f$ and $g$.  
More precisely, if $h=f\circ g$ is the function such that $h(x)=f(g(x))$ for every $x$.
In Lagrange's notation,
$$
h'(x) = f'(g(x)) g'(x)
$$
Equivently,
$$
h' = (f \circ g)' = (f' \circ g) \cdot g'.
$$
The chain rule may also be expressed in Leibniz's notation. If a variable $z$ depends on the variable $y$, which itself depends on the variable $x$ (that is, y and z are dependent variables), then $z$ depends on $x$ as well, via the intermediate variable $y$.  
In this case, the chain rule is expressed as,
$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx},
$$
and below expression for indicating at which points the derivatives have to be evaluated.
$$
\left. \frac{dz}{dx} \right|_{x} = \left. \frac{dz}{dy} \right|_{y(x)} \cdot \left. \frac{dy}{dx} \right|_{x}.
$$
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
https://www.geeksforgeeks.org/difference-between-differential-and-derivative/
https://math.stackexchange.com/questions/289923/what-are-the-differences-between-differential-and-gradient

## Epsilon in Calculus
- Epsilon is a small positive number, often used in numerical computation to avoid division by zero or taking the logarithm of zero.

## Epsilon-Delta Definition of Limit

## Taylor Series
The Taylor expansion of function is an infinite sum of terms that are expressed in terms of the function's derivatives at a single point. For most common functions, the function and the sum of its Taylor series are equal near this point.  
주어진 함수를 정의역의 특정 점의 미분계수들을 계수로 하는 다항식의 극한(멱급수)으로 표현하는 것을 말한다.
$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n
$$
간단히 설명하자면, 테일러 급수란 여러 번 미분이 가능한 함수 $f(x)$에 대해 
$x=a$ 에서 그 $f(x)$ 에 접하는 멱급수로 표현하는 방법이라고 할 수 있다.

$$
f(x) = p_{\infty}(x) \\
p_n(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x - a)^n \\
= \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x - a)^k  
$$

테일러 급수에서 주의해야 될 사항은 좌변과 우변이 모든 $x$에 대해 같은 것이 아니라 $x = a$ 근처에서만 성립한다는 점입니다. 즉, $x$가 $a$에서 멀어지면 멀어질수록 $f(x) = p(x)$로 놓는 것은 큰 오차를 갖게 됩니다. 한편, 근사다항식의 차수는 높으면 높을수록 $f(x)$를 좀더 잘 근사하게 됩니다.
테일러 급수는 결국 $x = a$에서 $f(x)$와 동일한 미분계수를 갖는 어떤 다항함수로 $f(x)$를 근사시키는 것입니다. 위 식에서
$$f(a) = p(a), f'(a) = p'(a), f''(a) = p''(a), ...$$
임은 쉽게 확인할 수 있을 것입니다. 테일러 급수를 이용해 이와같이 $x = a$ 에서 미분계수를 일치시키면 $x = a$ 뿐만 아니라 그 주변의 일정 구간에서도 $f(x)$ 와 $p(x)$ 가 거의 일치되게 됩니다.  
그런데 문제에 따라서는 $f(x)$ 를 1차 또는 2차까지만 테일러 전개하는 경우도 많습니다. 예를 들어, $f(x)$ 를 2차 다항함수로 근사할 경우에는
$$
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + Q_3(x)
$$
1차 다항함수로 근사할 경우에는
$$
f(x) = f(a) + f'(a)(x - a) + Q_2(x)
$$
와 같이 놓고 $Q(x)$를 0처럼 생각(무시)해 버립니다. 이 경우, $f(x)$를 무한차수 다항함수로 근사하는 것 보다는 근사오차가 크겠지만, $x$가 충분히 $a$에 가까운 경우에는 근사오차가 거의 없다고 볼 수 있습니다. 따라서 아래와 같이 표현할 수도 있습니다($x=a+h$). 
$$
f(a + h) = f(a) + f'(a) h + \frac{f''(a)}{2!} h^2 + \frac{f^{(3)}(a)}{3!} h^3 + \dots
= \sum_{k=0}^{\infty} \frac{f^{(k)}(a)}{k!} h^k
$$

### Using Taylor Series
When $f(x) = x^TAx$, using Taylor Series,
$$
f(x + h) = (x + h)^T A (x + h) \\
= x^T A x + x^T A h + h^T A x + h^T A h \\
= x^T A x + x^T (A + A^T) h + h^T A h \\
= f(x) + Df(x) h + o(|h|),
$$
So $\frac{\partial}{\partial x} \left (x^T A x \right)$ is $(A + A^T) x$ if $A$ is not symmetric, and $2Ax$ if $A$ is symmetric because $A^T = A$ when $A$ is symmetric matrix.


## Integral

### Leibniz Integral Rule
Leibniz Integral Rule is for differentiation under the integral sign.  
When $-\infty <a(x),b(x)<\infty$ and the integrands $a(x), b(x)$ are functions dependent on $x$, the derivative of this integral is expressible as below.
$$
\frac{d}{dx}
\Biggl(\int_{a(x)}^{\,b(x)} f\bigl(x,t\bigr)\,dt\Biggr)
= \\
f\bigl(x, b(x)\bigr)\,\frac{d}{dx}\,b(x)
\;-\;
f\bigl(x, a(x)\bigr)\,\frac{d}{dx}\,a(x)
\;+\;
\int_{a(x)}^{\,b(x)} \frac{\partial}{\partial x}f\bigl(x,t\bigr)\,dt
$$

In the special case where the functions $a(x)$ and $b(x)$ are constants $a(x)=a$ and $b(x)=b$ with values that do not depend on $x$, above equation can be  simplified as below.
(Simply, under standard regularity conditions (everything is nicely continuous/differentiable), you can swap the order of “take gradient w.r.t. $\theta$" and “integrate w.r.t. $y$".  
$$
{\frac {d}{dx}}\left(\int _{a}^{b}f(x,t)\,dt\right)=\int _{a}^{b}{\frac {\partial }{\partial x}}f(x,t)\,dt
$$

If $a(x)=a$ is constant and $b(x)=x$, which is another common situation, the Leibniz integral rule becomes:
$$
{\frac {d}{dx}}\left(\int _{a}^{x}f(x,t)\,dt\right)=f{\big (}x,x{\big )}+\int _{a}^{x}{\frac {\partial }{\partial x}}f(x,t)\,dt
$$

