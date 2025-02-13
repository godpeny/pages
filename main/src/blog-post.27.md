# Independent Components Analysis (ICA)
## Preliminaries
### CDF and PDF
### Density and Probability
### Permutation Matrix
A permutation matrix is a square binary matrix that has exactly one entry of 1 in each row and each column with all other entries 0.
For example,
$$
P =
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix},
\quad
v =
\begin{bmatrix}
v_1 \\ v_2 \\ v_3
\end{bmatrix}
$$
$Pv$ is as following.
$$
Pv =
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\ v_2 \\ v_3
\end{bmatrix}
=
\begin{bmatrix}
v_2 \\ v_3 \\ v_1
\end{bmatrix}
$$
#### Property of Permutation Matrix
- The inverse of a permutation matrix($P^{-1}$) is also a permutation matrix.
- The inverse is simply the transpose: $P^T = P^{-1}$

This holds for all permutation matrices, regardless of size.

### Rotatinal Matrix
### Rotational Symmetric


## Basics
Independent Component Analysis attempts to decompose a multivariate signal into independent non-Gaussian signals. It aims to find a linear transformation of data that maximizes statistical independence among the components.  
ICA is widely applied in fields like audio, image processing, and biomedical signal analysis to isolate distinct sources from mixed signals.

### Cocktail Party Problem

## Ambiguities of ICA

## Algorithm