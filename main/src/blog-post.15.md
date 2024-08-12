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
$
\mathbf{a} = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix}.  
$  

Column Vector is a $ m \times 1$ matrix for some $m$ consisting of a single column of ⁠$m$ entries,
$
\mathbf{x} = \begin{bmatrix} 
x_1 \\ 
x_2 \\ 
\vdots \\ 
x_m 
\end{bmatrix}.
$

### Trace
The trace of a square matrix $A$, denoted $\text{tr}(A)$, is defined to be the sum of elements on the main diagonal (from the upper left to the lower right) of matrix $A$.  
The trace is only defined for a square matrix $(n \times n)$.  
$
\text{tr}(A) = \sum_{i=1}^{n} a_{ii} = a_{11} + a_{22} + \cdots + a_{nn}
$  

### Span, Range, Rank and Null Space
### Vector Multiplication
https://rfriend.tistory.com/145
https://rfriend.tistory.com/146
- Dot Product
- Inner Product
- Outer Product
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
An affine function is a function composed of a linear function + a constant and its graph is a straight line. (즉 가중치 합(=Weighted Sum)에 bias(b)를 더해준 것)
## Affine hull
smallest affine set containing S, in other words, the intersection of all affine sets containing S.

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
 