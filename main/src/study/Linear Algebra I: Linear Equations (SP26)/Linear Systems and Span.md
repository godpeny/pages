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
### Echelon Form and RREF
#### Echelon Form
A rectangular matrix is in echelon form if,
1. All zero rows (if any are present) are at the bottom.
2. The first non-zero entry (or leading entry) of a row is to the right of any leading entries in the row above it (if any).
3. All entries below a leading entry (if any) are zero.

For example, matrix A is in echelon form while B is not in echelon form.
$$
A = 
\begin{pmatrix}
2 & 0 & 1 & 1 \\
0 & 0 & 5 & 3 \\
0 & 0 & 0 & 0 \\
\end{pmatrix} \quad 
B = 
\begin{pmatrix}
0 & 0 & 3 \\
0 & 0 & 2 \\
\end{pmatrix}
$$
In matrix A, 2 and 5 are leading entries and below leading entries are all 0s. However, in Matrix B, 2 is not leading entry.

#### Row Reduced Echelon Form (RREF)
A matrix in echelon form is in row reduced echelon form (RREF) if
1. All leading entries, if any, are equal to 1.
2. Leading entries are the only nonzero entry in their respective column.

For example, matrix A is RREF while B is not RREF and only EF.
$$
A = 
\begin{pmatrix}
1 & 0 & 0 & 1 \\
0 & 0 & 1 & 3 \\
0 & 0 & 0 & 0 \\
\end{pmatrix} \quad 
B = 
\begin{pmatrix}
1 & 0 & 6 & 1 \\
0 & 0 & 1 & 3 \\
0 & 0 & 0 & 0 \\
\end{pmatrix}
$$

In matrix A, 1 are leading entries in first and second rows and they are the only nonzero entry in their respective column. However, in Matrix B, there is 6 in first row above the leading entry of second row.

### Row Reduction Algorithm
In mathematics, Gaussian elimination, also known as row reduction, is an algorithm for solving systems of linear equations. It consists of a sequence of row-wise operations performed on the corresponding matrix of coefficients.  

To perform row reduction on a matrix, one uses a sequence of elementary row operations to modify the matrix until the lower left-hand corner of the matrix is filled with zeros, as much as possible. There are three types of elementary row operations:

1. Swapping two rows,
2. Multiplying a row by a nonzero number, and
3. adding a multiple of one row to another row.

Using these operations, a matrix can always be transformed into reduced row echelon form(RREF). 

### Pivot Position and Pivot Column
A pivot position in a matrix is the location of a leading entry (the first non-zero term) in a row of a matrix when it is in row echelon form (REF) or reduced row echelon form (RREF).

A pivot column is a column of A that contains a pivot position.

### Basic and Free Variables
$$
\left( A \mid \vec{b} \right) =
\left(
\begin{array}{ccccc|c}
1 & 0 & 0 & 7 & 2 & 4 \\
0 & 0 & 1 & 4 & 4 & 5 \\
0 & 0 & 0 & 0 & 2 & 4
\end{array}
\right)
$$
- Leading entries(pivots): in first, third, and fifth columns.
- Pivot columns: in the first, third, and fifth columns
- Variables that correspond to pivots are $x_1$, $x_3$, and $x_5$.
- Variables that correspond to a pivot are basic variables.
- Variables that are not basic are free variables. They can take any value.
- The free variables are $x_2$ and $x_4$. Any choice of the free variables leads to a solution of the system.

If A has $n$ columns, then the linear system must have $n$ variables. And a variable cannot be both free and basic at the same time. Therefore, 
$$
n = \text{number of columns of A} = \text{number of basic variables} + \text{number of free variables}
$$

### Existence and Uniqueness
A linear system is consistent if and only if (exactly when) the last column of the augmented matrix does not have a pivot as follows.
$$
\left(
\begin{array}{cc|c}
1 & 0 & \frac{1}{2} \\
0 & 1 & \frac{1}{2}
\end{array}
\right)
$$
Note that the third=last column(1/2, 1/2) has no pivot.  

If a linear system is consistent,
1. a unique solution if and only if there are no free variables.
2. infinitely many solutions that are parameterized by free variables.



## Topic 3: Vector Equations