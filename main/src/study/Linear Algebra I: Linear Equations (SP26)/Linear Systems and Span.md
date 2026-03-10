# Linear Systems and Span

## Topic 1: Systems of Linear Equations, Incomplete
### Linear Equation
A linear equation has the form as follows.
$$
a_1x_1 + a_2x_2 + \dots + a_nx_n = b
$$
- $a_n, b$ is coefficient.
- $x_n$ is variable.
- $n$ is number of variables = dimensions.

### Systems of Linear Equations
When we have one or more linear equation, we have a linear system of
equations.
$$
a_1x_1 + a_2x_2 + \dots + a_nx_n = c \\[5pt]
b_1x_1 + b_2x_2 + \dots + b_nx_n = d
$$

The set of all possible values of $x_1, x_2, \dots, x_n$ that satisfy all equations is the solution set of the system. One point in the solution set is a solution.  

The number of solution set to a system of linear equations can be only following three.
1. exactly one point (there is a unique solution)
2. infinitely many points (there are many solutions)
3. no points (there are no solutions)

### Row Operations
To find the solution set to a set of linear equations, we manipulate equations in a linear system using row operations.

1. Replacement/Addition: Add a multiple of one equation to another.
2. Interchange: Interchange two equations.
3. Scaling: Multiply an equation by a non-zero scalar.

Note that applying these operations to a linear system we do not change the solution set.

### Adjacent Matrices
We can rewrite systems $x_1, x_2, \cdots, x_n$  using matrices.  
For example,
$$
x_1 - 2x_2 + x_3 = 0 \\[5pt]
\quad 2x_2 -8x_3 = 7
$$

can be written as the augmented matrix as follows.

$$
\begin{pmatrix}
1 & -2 & 1 & \mid & 0 \\
0 & 2 & -8 & \mid & 7
\end{pmatrix}
$$

The vertical line reminds us that the first three columns are the coefficients to our variables $x_1$, $x_2$, and $x_3$. Row operations can be applied to rows of augmented matrices as though they were coefficients in a system.

### Consistent Systems and Row Equivalence
- A linear system is consistent if it has at least one solution.
- Two matrices are row equivalent if a sequence of row operations transforms one matrix into the other.

## Topic 2: Row Reduction and Echelon Forms, Incomplete
## Topic 3: Vector Equations