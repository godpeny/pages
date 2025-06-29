# Independent Components Analysis (ICA)
## Preliminaries
### Density and Probability
Probability($\rho(x) \delta x $) is the computing the integral of probability density ($\rho(x)$) over a given interval($\delta x$).  
In simple terms, a probability density tells us how likely different values of a random variable are.
But unlike discrete probabilities, it does not directly give probabilities. Instead, we integrate the density function to find probabilities over intervals.
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

### Rotational Matrix
Rotation matrix is a transformation matrix that is used to perform a rotation in Euclidean space. 
$$
{\displaystyle R={\begin{bmatrix}\cos \theta &-\sin \theta \\\sin \theta &\cos \theta \end{bmatrix}}}
$$
For example, above matrix rotates points in the $xy$ plane counterclockwise through an angle $\theta$ about the origin of a two-dimensional Cartesian coordinate system.
 - ${\displaystyle R^{\mathsf {T}}R = {\displaystyle RR^{\mathsf {T}}} =  I}$
 - ${\displaystyle \det R=+1}$


### Rotational Symmetric
Rotational symmetry, also known as radial symmetry in geometry, is the property a shape has when it looks the same after some rotation by a partial turn.

## Basics
Independent Component Analysis attempts to decompose a multivariate signal into independent non-Gaussian signals. It aims to find a linear transformation of data that maximizes statistical independence among the components.  
ICA is widely applied in fields like audio, image processing, and biomedical signal analysis to isolate distinct sources from mixed signals.

### Cocktail Party Problem
The $n$ speakers are speaking simultaneously at a party, and any microphone placed in the room records only an overlapping combination of the $n$ speakers’ voices.  
But let’s say we have $n$ different microphones placed in the room, and because each microphone is a different distance from each of the speakers, it records a different combination of the speakers’ voices. Using these microphone recordings, can we separate out the original $n$ speakers’ speech signals?

## Definition of ICA
Assume that we observe $n$ linear mixtures $x_1, 
\cdots ,x_n$ of $n$ independent components. Since independent component $s_k$ is random variable, linear transformation of $s$ is $x$ and therefore $x$ is also random variable. But $x$ is just a mixture of independent components($=s$), but $x$ itself is not independent.
$$
x_j = \sum_{j}^m a_{j1}s_1 + a_{j2}s_2 + ... + a_{jn}s_n
$$
It is convenient to use vector-matrix notation instead of the sums like in the previous equation.  
Let us denote by $x$ the random vector whose elements are the mixtures $x_1, \cdots x_n$, and likewise by $s$ the random vector with elements $s_1, \cdots , s_n$. Let us denote by $A$ the matrix with elements $a_{ij}$. 

$$ x = As $$

The ICA model is a generative model, which means that it describes how the observed data are generated by a process of mixing the components $s_k$ when some data $s \in \mathbb{R}^n$ that is generated via $n$ independent sources.

So $s_k$'s are latent variables, meaning that they cannot be directly observed. Also the mixing matrix $A$ is assumed to be unknown.  
All we observe is the random vector $x$, and we must estimate both $A$ and $s$ using it.
There is  What we observe is $x$, where $A$ is an unknown square matrix called the mixing matrix.  

Therefore, our goal is to recover the sources $s$ that had generated our data. In other words, after estimating the matrix $A$, we can compute its inverse, say $W$($W = A^{-1}$), and obtain the independent component simply by:
$$
{\bf s}={\bf W}{\bf x}.
$$

## Ambiguities of ICA
It is easy to see that the following ambiguities will hold.

### Scalar of the Sources($A, s$)  
We cannot determine the correct scaler of the sources($A, s$).  
The reason is that, both $s$ and $A$ being unknown, any scalar multiplier in one of the sources $s_k$ could always be cancelled by dividing the corresponding column $a_j$ of by the same scalar.  
For example, if $A$ were replaced with $2A$ and every $s_k$ were replaced with $(0.5)s_k$,observed $x_k=2A·(0.5)s_k$ would still be the same.

### Order of the Sources($A, s$)  
We cannot determine the order of the sources($A, s$).  
The reason is that, again both $s$ and $A$ being unknown, we can freely change the order of the terms in the sum.  
For example, below is the case when $S_{\text{estimated}}$ is correctly recovered.
$$
X = AS =
\begin{bmatrix} 
2 & 1 \\ 
1 & 3 
\end{bmatrix}
\begin{bmatrix} 
1 & 2 \\ 
3 & 4 
\end{bmatrix}  =
\begin{bmatrix}
5 & 8 \\
10 & 14
\end{bmatrix}
$$
$$
W = A^{-1} =
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}^{-1}
=
\begin{bmatrix}
\frac{3}{5} & -\frac{1}{5} \\
-\frac{1}{5} & \frac{2}{5}
\end{bmatrix}
$$
$$
S_{\text{estimated}} = WX = 
\begin{bmatrix}
\frac{3}{5} & -\frac{1}{5} \\
-\frac{1}{5} & \frac{2}{5}
\end{bmatrix}
\begin{bmatrix}
5 & 8 \\
10 & 14
\end{bmatrix}
=
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$
However, with permutation matrix $P = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$, 
$$
\tilde{W} = PW
$$
$$
\tilde{W} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
\frac{3}{5} & -\frac{1}{5} \\
-\frac{1}{5} & \frac{2}{5}
\end{bmatrix}
=
\begin{bmatrix}
-\frac{1}{5} & \frac{2}{5} \\
\frac{3}{5} & -\frac{1}{5}
\end{bmatrix}
$$
$$
S_{\text{estimated, permuted}} = \tilde{W} X = 
\begin{bmatrix}
-\frac{1}{5} & \frac{2}{5} \\
\frac{3}{5} & -\frac{1}{5}
\end{bmatrix}
\begin{bmatrix}
5 & 8 \\
10 & 14
\end{bmatrix}
=
\begin{bmatrix}
3 & 4 \\
1 & 2
\end{bmatrix}
$$
The estimated sources are permuted but still correct in terms of independence. (column space = span remains unchanged)

### Ambiguities in practice
However, for the applications that we are concerned with(including cocktail problem) this ambiguity also does not matter.  
Specifically, scaling a speaker’s speech signal $s_k^{(i)}$ j by some positive factor $\alpha$ affects only the volume of the sound.  
Also, sign changes do not matter,and $s_k^{(i)}$ and $−s_k^{(i)}$ sound identical when played on a speaker. This is also true for the ICA for the brain/MEG data.

### Why $s$ has to be Non-Gaussian?
It turns out that these two are the only ambuguities so long as the sources $s$ are non-Gaussian.   
Let's see what the difficulty is with Gaussian data. But before move on, you should know several notes as below.

 - Note 1: In ICA, the source signals $s$ are typically assumed to have zero mean and unit variance (variance = 1). So technically $s$ has to be standard normal distribution. If $s$ is normally distributed, using Whitening transformation to make it standard distribution.
 - Note 2: Also Note that the contours of the density of the standard normal distribution $N(0,I)$ are circles centered on the origin, and the density is rotationally symmetric. 
 - Note 3: A multivariate Gaussian distribution(which is $s$ in this case) has a spherical symmetry in high dimensions. This means that any orthogonal rotation of $s$ will still be Gaussian.

Using above notes, let $s \sim \text{N}(0,I)$ is independent variable with zero mean and unit variance. Consier observed value $x = As$,
$$
\mathbb{E}[xx^T] = \mathbb{E}[A s s^T A^T] = A A^T.
$$ 
Where $A$ is our mixing matrix.
Above equation can be derived because covariance of $s$ can be calculated as, 
$$
\text{Cov}(s) = \mathbb{E} \left[ (s - \mathbb{E}[s])(s - \mathbb{E}[s])^T \right] = \mathbb{E}[s s^T]
$$
Since $\mathbb{E}[s] = 0$ from given condition.

Now Let's think of rotational matrix $R$ and apply to mixing matrix $A$. When $A' = AR$, Then if the data $x$ had been mixed according to $A$ would have instead observed as $x' = A's$.  
The distribution of $x'$ is also Gaussian, with zero mean and covariance $AA^T$.  
This is because mean can be calcuated as 
$$\mathbb{E}[x'] = A \mathbb{E}[s]$$ 
When $\mathbb{E}[s] = 0$.  

Also variance is 
$$\mathbb{E}[x' (x')^T] = \mathbb{E}[A' s s^T (A')^T] = \mathbb{E}[A R S S^T (A R)^T] = A R R^T A^T = A A^T
$$

Therefore, whether the mixing matrix is $A$ or $A'$, we would observe data from $x \sim \text{N}(0,AA^T)$ distribution.  
So, there is no way to tell if the sources were mixed using $A$ and $A'$.
If there is an arbitrary rotational component $R$ in the mixing matrix that cannot be determined from the data, we cannot recover the original sources.  
In other words, as long as the data is not Gaussian, it is possible to recover the $n$ independent sources.

### Why Permutation/Sign/Scale is okay but Rotation is not?
 - Permutation ± sign ± scale: no cross-mixing, harmless for most applications.
 - Rotation: mixes rows/columns, prevents source recovery unless the non-Gaussianity criterion can single out the true unmixing rotation.

For example, consider below true sources.
$$
s_{1}(t)=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
s_{2}(t)=\begin{bmatrix}0\\1\end{bmatrix}
$$
$A$ is mixing matrix and therefore $x$ is observed mixtures.
$$
A=\begin{bmatrix}
2 & 1\\
1 & 1.5
\end{bmatrix},
\qquad
x = A\,s \\[6pt]
$$
When rotation matrix $R$ is below.
$$
R=\frac{1}{\sqrt{2}}
\begin{bmatrix}
 1 & -1\\
 1 & \phantom{-}1
\end{bmatrix}
$$
$$
AR  = \frac{1}{\sqrt{2}}
      \begin{bmatrix}
        3 & -1\\[4pt]
        2.5 & 0.5
      \end{bmatrix} \\[6pt]
x_{\text{rot}}
      = (AR)s
      = \frac{1}{\sqrt{2}}
        \begin{bmatrix}
          3\,s_{1} - 1\,s_{2}\\[6pt]
          2.5\,s_{1} + 0.5\,s_{2}
        \end{bmatrix}.
$$
You can see that column of $AR$ is not a scaled copy of either original column of $A$. It is already a linear combination of the two.

When permuation matrix $P$ and scaling and singing matrix $D$ is each below.
$$
P =
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix},
\qquad
D =
\begin{bmatrix}
1 & 0\\
0 & -3
\end{bmatrix},
$$
$$
APD = \begin{bmatrix}
        1 & -6\\[4pt]
        1.5 & -3
      \end{bmatrix} \\[6pt]
x_{\text{pss}}
      = (APD)s
      = \begin{bmatrix}
          1\,s_{2} - 6\,s_{1}\\[6pt]
          1.5\,s_{2} - 3\,s_{1}
        \end{bmatrix}
$$
However, when you see matrix $APD$, it's first column is the original second column of $A$ and second column is $-3$ scaled first column of $A$.


## Linear Transformations on Densities
If $s$ is a vector-valued distribution with density $p_s$, and $x =As$ for a square and invertible matrix $A$, then the density of $x$ is given by,
$$
p_x(x) = p_s(Wx) \cdot |\det W|,
$$
Where $W = A^{-1}$.

For example, Let's say mixing matrix $A$ and unmixing matrix $W$ as below.
$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}, \quad
|A| = ad - bc
\\
W = \frac{1}{ad - bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}, \quad
|W| = \left| \frac{\det A}{(\det A)^2} \right| = \frac{1}{ad - bc}
$$
When $x=As$, $x$ can be considered as the linear transformed vector $s$ with matrix $A$. Since PDF always integrates to 1, the area of the parallelogram formed by the vectors has to be also 1.   
![alt text](images/blog27_linear_transformation_of_densities.png)  
Since the area of the parallelogram is the absolute value of the determinant of the matrix formed by the vectors, which is $ad-bc$, scaling with $|W|$ makes the area always 1.

## Algorithm
Since the distribution of each source $s_k$ is given by a density $p_s$, 
the joint distribution of the sources $s_k$ is given by,
$$
p(s) = \prod_{k=1}^{n} p_s(s_k).
$$
Assuming that the sources are independent.  
Note that the joint distribution $p(s)$ represents the product of the individual source distributions. In other words, it is probability of all independent source variables($s_k$) occurring together.

Using the formulas $p_x(x) = p_s(Wx) \cdot |\det W|$ from "Linear Transformations on Densities" let's find out the density on $x = A s = W^{-1} s$.
$$
p(x) = \prod_{i=1}^{n} p_s(w_i^T x) \cdot |\det W| = p_s(Wx) \cdot |\det W|
$$

Now, let's specify a density for the individual sources $p_s$ from above $p(x)$.
To specify a density for the $s_k$’s, all we need to do is to specify some cdf for it. Following our previous discussion("Why $s$ has to be Non-Gaussian?"), we cannot choose cdf of the Gaussian as cdf of $s_k$, as ICA doesn’t work on Gaussian data.  
What we’ll choose as a reasonable cdf is the sigmoid function $g(s) = \frac{1}{1 + e^{-s}}$. Hence, $p_s(s) = g'(s)$, since CDF $F(s) = g(s)$.
The reason why this works is that the sigmoid function satisfies the properties of a cumulative distribution function(monotonic function that increases from zero to one), and by differentiating such a function, we get a probability density function.

Now, keep in mind that square matrix $W$ is the parameter for the model, we use MLE to find out the matrix $W$ that maximize the log liklihood. Which is,
$$
\ell(W) = \sum_{i=1}^{m} \left( \sum_{j=1}^{n} \log g'(w_j^T x^{(i)}) + \log |W| \right)
$$
By taking derivatives of log likelihood $\nabla_W \ell(W)$ and using the fact that $\nabla_W |W| = |W| (W^{-1})^T$, we easily derive a stochastic gradient ascent learning rule as follow.
$$
W := W + \alpha \left( 
\begin{bmatrix}
1 - 2g(w_1^T x^{(i)}) \\
1 - 2g(w_2^T x^{(i)}) \\
\vdots \\
1 - 2g(w_n^T x^{(i)})
\end{bmatrix} 
x^{(i)T} + (W^T)^{-1}
\right)
$$
Where $\alpha$ is learning rate.
After the algorithm converges, compute $s^{(i)} = Wx^{(i)}$ to recover the original sources $s$.  
Mathmatic derivation of gradient of log sigmod function is as below.
$$
g(u)=\sigma(u)=\frac{1}{1+e^{-u}}
\;\;\Longrightarrow\;\;
g'(u)=\sigma(u)\bigl(1-\sigma(u)\bigr) \\[6pt]
\log g'(u)
=\log\!\bigl[\sigma(u)\bigl(1-\sigma(u)\bigr)\bigr]
=\underbrace{\log\sigma(u)}_{\text{term A}}
+\underbrace{\log\!\bigl(1-\sigma(u)\bigr)}_{\text{term B}} \\[6pt]
\text{Term A:}\quad
\frac{d}{du}\log\sigma(u)
=\frac{\sigma'(u)}{\sigma(u)}
=\frac{\sigma(u)\bigl(1-\sigma(u)\bigr)}{\sigma(u)}
=1-\sigma(u).
\\[6pt]
\text{Term B:}\quad
\frac{d}{du}\log\!\bigl(1-\sigma(u)\bigr)
=-\frac{\sigma'(u)}{1-\sigma(u)}
=-\,\frac{\sigma(u)\bigl(1-\sigma(u)\bigr)}{1-\sigma(u)}
=-\sigma(u).
\\[6pt]
\frac{d}{du}\log g'(u)
=\bigl(1-\sigma(u)\bigr)-\sigma(u)
=\boxed{\,1-2\sigma(u)\,}.
$$
Also, Since $(W^{\top})^{-1} \;=\; \bigl(W^{-1}\bigr)^{\!\top}$ and $\nabla_W |W| = |W| (W^{-1})^T$, mathmatic derivation of gradient  log determinant of $W$ can be expressed as below.
$$
\nabla_{W}\,\log\bigl|W| = \bigl(W^{-1}\bigr)^{\!\top}
$$