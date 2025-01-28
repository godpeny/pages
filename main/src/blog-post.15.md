# Linear Algebra
## Feasible Point/Region
 - point : A point x that satisfies all the constraints is called a feasible point and thus is a feasible solution to the problem. 
 - region : A set of all possible points (sets of values of the choice variables) of an optimization problem that satisfy the problem's constraints, potentially including inequalities, equalities, and integer constraints.
## Convex and Concave Function
Convex function : function is convex if the line segment between any two distinct points on the graph of the function lies above the graph between the two points. (볼록 함수 - 아래로 볼록)
Concave function : function is concave if the line segment between any two distinct points on the graph of the function lies below the graph between the two points. (오목 함수 - 아래로 오목)
## Convex Optimization
- Convex optimization problems : subfield of mathematical optimization that studies the problem of minimizing convex functions over convex sets (or, equivalently, maximizing concave functions over convex sets).
- Convex set : given any two points in the subset, the subset contains the whole line segment that joins them. 
  Equivalently, a convex set or a convex region is a subset that intersects every line into a single line segment.
- Optima in Convex Optimization : local optima are global optima.

## Domain of a Function
- the set of inputs accepted by the function. It is sometimes denoted by dom(f) or dom f,  where f is the function. 
- in layman's term, "what x can be".
- e.g. ``f(X) -> y, dom(f) = X``

## Vertical Bars
- $|x|$ if $x$ is a number, it denotes abolute value.  
- $|A|$  if $A$ is a matrix, it denotes determinant.  
- $|S|$ if $S$ is a set, it denotes its cardinality(the number of elements of the set).

## Vector and Matrix
### Row Vector vs Column Vector
Row Vector is a $1 \times n$ matrix for some $n$, consisting of a single row of $n$ entries,
$$
\mathbf{a} = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix}.  
$$

Column Vector is a $ m \times 1$ matrix for some $m$ consisting of a single column of ⁠$m$ entries,
$$
\mathbf{x} = \begin{bmatrix} 
x_1 \\ 
x_2 \\ 
\vdots \\ 
x_m 
\end{bmatrix}.
$$

### Vector Double Summation
$$
\begin{align*}
\left( \sum_{i=1}^n x_i \right)^2 &= \left( \sum_{i=1}^n x_i \right) \left( \sum_{j=1}^n x_j \right) = \sum_{i=1}^n \sum_{j=1}^n x_i x_j
\end{align*}
$$

### Trace
The trace of a square matrix $A$, denoted $\text{tr}(A)$, is defined to be the sum of elements on the main diagonal (from the upper left to the lower right) of matrix $A$.  
The trace is only defined for a square matrix $(n \times n)$.  
$
\text{tr}(A) = \sum_{i=1}^{n} a_{ii} = a_{11} + a_{22} + \cdots + a_{nn}
$  

### Vector and Identity Matrix
When $z$ is a vector of dimension of $n \times 1$ and matrix $H=a⋅I$, where $I$ is identity matrix of $n \times n$ dimension and $a$ is scalar.
$$
z \cdot (a \cdot I) \cdot z^T = a \cdot (z \cdot I \cdot z^T) = a \cdot (z \cdot z^T)
$$
It is shwon that multiplying by the identity matrix $I$ does not change the vector.  
Also, it can be presented as below:  
$$
\quad z \cdot H \cdot z^T = a \cdot \|z\|^2,
\quad \text{where} \quad \|z\|^2 = z \cdot z^T = \sum_{i=1}^{n} z_i^2.
$$

### Vector Space (Linear Space)
A vector space is a set whose vectors can be added together and multiplied ("scaled") by numbers called scalars.  
The operations of vector addition and scalar multiplication must satisfy certain requirements, called vector axioms.
![alt text](images/blog15_vector_space.png)  
For example, above image show the vector addition and scalar multiplication.
 - 1) a vector $v$ (blue) is added to another vector $w$ (red, upper illustration). 
 - 3) $w$ is stretched by a factor of $2$, yielding the sum $v + 2w$.

### Range

### Null Space

### Span (Linear Span)
The linear span of a set $S$ of elements of a vector space $V$ is the smallest linear subspace of 
$V$ that contains $S$.  
It is the set of all finite linear combinations of the elements of $S$, and the intersection of all linear subspaces that contain $S$.  
It follows from this definition that the span of $S$ is the set of all finite linear combinations of elements (vectors) of $S$, and can be defined as following,
$$
\text{span}(S) = \left\{ \lambda_1 \mathbf{v}_1 + \lambda_2 \mathbf{v}_2 + \cdots + \lambda_n \mathbf{v}_n \mid n \in \mathbb{N}, \mathbf{v}_1, \dots, \mathbf{v}_n \in S, \lambda_1, \dots, \lambda_n \in K \right\}
$$
For example, suppose there is vectore space $\mathbb{R}^{3}$,
$$
{(1, 0, 0), (0, 1, 0), (1, 1, 0)} 
$$
However, its spanning set is not ${(1, 0, 0), (0, 1, 0), (1, 1, 0)}$. Instead, its spanning set is, 
$$
{(1, 0, 0), (0, 1, 0)} 
$$
This is because last component $(1, 1, 0)$ is a linear combination of $(1, 0, 0)$ and $(0, 1, 0)$. Thus, the spanned space is not $\mathbb{R}^{3}$.

### Rank
The rank of a matrix $A$ is the dimension of the vector space generated (or spanned) by its columns.  
This corresponds to the maximal number of linearly independent columns of $A$. This, in turn, is identical to the dimension of the vector space spanned by its rows. In other words,  
$\text{Row Rank} = \text{Column Rank} = \text{Rank of the Matrix}$

For example, below matrix has rank 2.
$$
(1, 0, 1) \\ (0, 1, 1)\\(0, 1, 1)
$$
 - Column Rank: first two column vectors are linearly independable, but third column can be attained by lienar combination of the first two($1+2$). So rank is 2.
 - Row Rank: second and third row vectors are identical, so rank is 2.

#### Property of Matrix Rank
 - Sylvester’s rank inequality: if $A$ is an $m \times n$ matrix and $B$ is $n \times k$, then,
 $$ \operatorname {rank} (A)+\operatorname {rank} (B)-n\leq \operatorname {rank} (AB) $$
 - Subadditivity: $$ \operatorname {rank} (A+B)\leq \operatorname {rank} (A)+\operatorname {rank} (B) $$

### Vector Multiplication
https://rfriend.tistory.com/145
https://rfriend.tistory.com/146

### Inner Product (Dot Product)
Inner Product also called as a scalar product is an algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors), and returns a single number.

#### Coordinate Definition
The dot product of two vectors $\mathbf {a} =[a_{1},a_{2},\cdots ,a_{n}]$ and $ \mathbf {b} =[b_{1},b_{2},\cdots ,b_{n}]$, specified with respect to an orthonormal basis, is defined as
$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n.
$$

For example, when there are two vectors, inner product of these vectors is,
$$
[1,3,-5] \cdot [4,-2,-1] = (1 \times 4) + (3 \times -2) + (-5 \times -1)
= 4 - 6 + 5 = 3
$$

#### Geometric Definition
In Euclidean space, a Euclidean vector is a geometric object that possesses both a magnitude(length of vector) and a direction. A vector can be pictured as an arrow.  
The dot product of two Euclidean vectors $\mathbf {a}$ and $\mathbf {b}$ is defined by,
$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta,
$$
When $\|\mathbf{a}\|$ denotes for magnitude(lenght) of vecdtor.  
If vecdtor $\mathbf{a}$ and $\mathbf{b}$ are orthogonal, (the angle is $ 90^\circ$, which mneans $\theta = \pi / 2$).  
Since $\cos \pi / 2 = 0$, 
$$
mathbf{a} \cdot \mathbf{b} = 0
$$


### Outer Product
Outer product of two coordinate vectors is the matrix whose entries are all products of an element in the first vector with an element in the second vector.  
If the two coordinate vectors have dimensions $n$ and $m$, then their outer product is an $n \times m$ matrix. 

#### Definition
Given two vectors of size $m\times 1$ and $n\times 1$ respectively,
$$
\mathbf{u} =
\begin{bmatrix}
    u_1 \\
    u_2 \\
    \vdots \\
    u_m
\end{bmatrix}, \quad
\mathbf{v} =
\begin{bmatrix}
    v_1 \\
    v_2 \\
    \vdots \\
    v_n
\end{bmatrix}
$$
The outer product of these two vectors is $m \times n$ matrix $\mathbf{A}$, obtained by multiplying each element of each vector.
$$
\mathbf{u} \otimes \mathbf{v} = \mathbf{A} =
\begin{bmatrix}
    u_1 v_1 & u_1 v_2 & \cdots & u_1 v_n \\
    u_2 v_1 & u_2 v_2 & \cdots & u_2 v_n \\
    \vdots & \vdots & \ddots & \vdots \\
    u_m v_1 & u_m v_2 & \cdots & u_m v_n
\end{bmatrix}
$$
As you can see from above equation, outer product generates the matrix whose first row is $u_1(v_1, v_2, \cdots, v_n)$ and the $i$-th row is $u_i(v_1, v_2, \cdots, v_n)$.  
So the rows are the vector $(v_1, v_2, \cdots, v_n)$ multiplied by scalars. Since this itself is the basis, the rank of the result matrix is always 1(or a zero matrix if the vector is zero).


#### Summing the result matrix from outer product.
Consider following sum of matrices of outer product of vectors.
$$
X = (X_1 + X_2 + \cdots + X_N) = u_1 v_1^T + u_2 v_2^T + \dots + u_N v_N^T 
$$
When $(u_1, u_2, \dots, u_N)$ are linearly independent and also $(v_1, v_2, \dots, v_N)$ is linearly independent.  
In this summation, rank of matrix $X$ is equal to $N$.
Let's use 2 properties of matrix rank, Subadditivity and Sylvester’s rank inequality to prove it.  
First, using Subadditivity, we can assure that $\operatorname {rank} (X) \leq N$.
$$ \operatorname {rank} (X) = \operatorname {rank} (X_1 + X_2 + \cdots + X_N) \leq (1 + 1 + \cdots + 1) = N $$
Second, consider $X$ as multiplication of two matrices and apply Sylvester’s rank inequality.
$$
X = U \cdot V = \begin{pmatrix} u_1 & \dots & u_N \end{pmatrix} 
\begin{pmatrix} v_1^T \\ \vdots \\ v_N^T \end{pmatrix} 
$$
$\operatorname{rank}(U)$ and $\operatorname{rank}(U)$ are both $N$ because they are composed of $N$'s linearly independent vectors from the first place.
$$
N(N + N - N) = \operatorname{rank}(U) + \operatorname{rank}(V) - N \leq \operatorname{rank}(X).
$$

Therefore, combining the two derivation above, we can conclude that the rank of sum of $N$ rank-1 matrices are $N$.

### Diagonal Matrix
In linear algebra, a diagonal matrix is a matrix in which the entries outside the main diagonal are all zero; the term usually refers to square matrices. Elements of the main diagonal can either be zero or nonzero.  
In geometry, a diagonal matrix may be used as a scaling matrix, since matrix multiplication with it results in changing scale (size) and possibly also shape; only a scalar matrix results in uniform change in scale.
#### Identity Matrix
An identity matrix of size $n$ is  $n\times n$ square matrix with ones on the main diagonal and zeros elsewhere. 
#### Scaling Matrix
To scale an object by a vector $\mathbf{v} = (v_x, v_y, v_z)$, each point $\mathbf{p} = (p_x, p_y, p_z)$ would need to be multiplied with this scaling matrix:

$
S_v = \begin{bmatrix}
v_x & 0 & 0 \\
0 & v_y & 0 \\
0 & 0 & v_z
\end{bmatrix}.
$

$
S_v p = \begin{bmatrix}
v_x & 0 & 0 \\
0 & v_y & 0 \\
0 & 0 & v_z
\end{bmatrix}
\begin{bmatrix}
p_x \\
p_y \\
p_z
\end{bmatrix}
= \begin{bmatrix}
v_x p_x \\
v_y p_y \\
v_z p_z
\end{bmatrix}.
$
#### Singular Matrix
A singular matrix is a square matrix whose determinant is zero. In other words, it’s a square matrix (where the number of rows and columns are equal) that has no inverse. 
In a singular matrix, some rows and columns are linearly dependent. Therefore, the rank of a singular matrix will be less than the order of the matrix, i.e., Rank (A) < Order of A.

### Matrix Multiplication
#### 3-d matrix multiplication
A 3D matrix is nothing but a collection (or a stack) of many 2D matrices, just like how a 2D matrix is a collection/stack of many 1D vectors.  
So, matrix multiplication of 3D matrices involves multiple multiplications of 2D matrices, which eventually boils down to a dot product between their row/column vectors.

```
Matrix A:
[[[1 2]
  [3 4]] // a_1

 [[5 6]
  [7 8]]] // a_2

Matrix B:
[[[9 8]
  [7 6]] // b_1

 [[5 4]
  [3 2]]] // b_2

Matrix C (A @ B):
[[[23 20] 
  [55 48]] // a_1 @ b_1

 [[47 36]
  [83 64]]] // a_2 @ b_2
```
## Affine Function
An affine function is a function composed of a linear function + a constant and its graph is a straight line.  
(즉 가중치 합(=Weighted Sum)에 bias(b)를 더해준 것)

## Affine Hull
The smallest affine set containing S, in other words, the intersection of all affine sets containing S.

## Affine Space
This is the set of points $x$ that satisfying,
$$
x = \sum_{i=1}^{m} \alpha_i x^{(i)}
$$
For some $\alpha_i$’s so that $\sum_{i=1}^{m} \alpha_i x^{(i)} = 1$

## Bounded/Unbounded
A set is called bounded if all of its points are within a certain distance of each other. 
Conversely, a set which is not bounded is called unbounded. 

## Bounded Above/Below, Upper/Lower Bound and Least Upper/Greast Lower Bound (Supremum, Infimum) 
- A set $E \subseteq \mathbb{R}$ is bounded above(or below) if there is a real number M, called an upper(or lower) bound of E, such that $x <= M$ (or $x >= M$), for all $x \in \mathbb{R}$

- A real number M is the least upper(or greast lower) bound, or supremum(or infimum), of a set  $E \subseteq \mathbb{R}$ if,
1. M is an upper(or lower) bound of E
2. each M' < M is not an upper bound of E. In this case, we write M = supE.  
(or each M' > M is not an lower bound of E. In this case, we write M = infE.)
- 쉽게 말하면, 상계(upper bound)에 속하는 값들 중에서 가장 작은 값이 상한(supremum)이 되고 하계(lower bound)에 속하는 값들 중에서 가장 큰 값이 하한(infimum)이 된다.

### Definite matrix
For real matrix and $n \times n$ symmetric matrix $\mathbf{M}$.  
An $n \times n$ symmetric real matrix $\mathbf{M}$ is said to be positive-definite if,
$$
\mathbf{x}^\top M \mathbf{x} > 0 \quad \text{for all non-zero } \mathbf{x} \in \mathbb{R}^n.
$$

Formally,
$$
M \text{ positive-definite } \iff \mathbf{x}^\top M \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n.
$$

An $n \times n$ symmetric real matrix $\mathbf{M}$ is said to be positive-semi definite if,
$$
\mathbf{x}^\top M \mathbf{x} \geq 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n.
$$
Formally,
$$
M \text{ positive-semidefinite } \iff \mathbf{x}^\top M \mathbf{x} \geq 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n.
$$

An $n \times n$ symmetric real matrix $\mathbf{M}$ is said to be negative-definite if,
$$
\mathbf{x}^\top M \mathbf{x} < 0 \quad \text{for all non-zero } \mathbf{x} \in \mathbb{R}^n.
$$
Formally,
$$
M \text{ negative-definite } \iff \mathbf{x}^\top M \mathbf{x} < 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n.
$$

An $n \times n$ symmetric real matrix $\mathbf{M}$ is said to be negative-semidefinite if,
$$
\mathbf{x}^\top M \mathbf{x} \leq 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n.
$$
Formally,
$$
M \text{ negative-semidefinite } \iff \mathbf{x}^\top M \mathbf{x} \leq 0 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n.
$$