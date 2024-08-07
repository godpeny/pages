# Statistics
## L2 Norm
The L2 Norm $|x|$ is a vector norm defined for a complex vector (A vector whose elements are complex numbers.)
when 
$
\mathbf{x} = \begin{bmatrix} 
x_1 \\ 
x_2 \\ 
\vdots \\ 
x_n 
\end{bmatrix}, \\
\|\mathbf{x}\| = \sqrt{\sum_{k=1}^n |x_k|^2},
$

## "Entry-wise" Matrix Norms
Treat $m\times n$  matrix as a vector of size $m\cdot n$ and use one of the familiar vector norms.  
$
\|\mathbf{A}\|_{p,p} = \|\operatorname{vec}(\mathbf{A})\|_p = \left( \sum_{i=1}^m \sum_{j=1}^n \left|a_{ij}\right|^p \right)^{1/p}
$  
The special case, p = 2 is the Frobenius norm, and p = ∞ yields the maximum norm.  
### Lpq Norm
From the original definition matrix, $A$ presents n data points in m-dimensional space.  
Since $(a_{1},\ldots ,a_{n})$ be the columns of matrix 
$\displaystyle L_{2,1}$ norm is the sum of the Euclidean norms of the columns of the matrix:  
$
\|\mathbf{A}\|_{2,1} = \sum_{j=1}^n \|\mathbf{a}_j\|_2 = \sum_{j=1}^n \left( \sum_{i=1}^m \left|a_{ij}\right|^2 \right)^{1/2}
$  
In general, when $p, q \geq 1$,  
$
\|\mathbf{A}\|_{p,q} = \left( \sum_{j=1}^n \left( \sum_{i=1}^m \left|a_{ij}\right|^p \right)^{q/p} \right)^{1/q}.
$
### Frobenius Norm
The Frobenius norm is an extension of the Euclidean norm to $K^{n \times n}$.  
When $p$ = $q$ = 2 for the $L_{p,q}$ it is called the Frobenius norm or the Hilbert–Schmidt norm.  
$
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2} = \sqrt{\text{trace}(A^* A)}.
$  
Loosely speaking, the Frobenius Norm is also equivalent to the Euclidean norm generalised to matrices instead of vectors.
## Gaussian Distribution (= Normal Distribution)
## Joint Probability
- P(A, B) = p(A ∩ B) : likelihood of events occurring together at the same point in time.
## Probability vs Likelihood
- In Deep Learning
  - same function can be interpreted in two ways: as a function of data given parameters, or as a function of parameters given data. 
  - probability : function of data given parameters. (parameters are fixed and data is varying)
  - likelihood : function of parameters given data. (data is fixed and parameters are varying)
- In Statistics
  - probability : chance that a particular outcome occurs based on the values of parameters in a model. 
    - (확률 분포가 고정된 상태에서, 관측되는 사건이 변화될 때 확률)
  - likelihood : how probable a particular set of observations is, given a set of model parameters. = likelihood is used to estimate the parameters of a model given a set of observations.
    - (관측된 사건이 고정된 상태에서, 확률 분포가 변화될 때(=확률 분포를 모를 때 = 가정할 때) 확률)
    - maximum likelihood estimation (MLE) = method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of observing the data.
  - In discrete case : probability = likelihood
  - In PDF
    - probability = area under the curve.
    - likelihood = y-axis value.

## Cumulative Distribution Function(CDF) vs Probability Density Function(PDF) vs Probability Mass Function(PMF)
### PDF: continuous random variable
![alt text](images/blog5_pdf.png)
#### Necessary Conditions for PDF
 - $f(x) \geq 0, \, \forall \, x \in \mathbb{R}$
 - $f(x)$ should be piecewise continuous.
 - $\int_{-\infty}^{\infty} f(x) \, dx = 1$

#### Properties of PDF
 - $\Pr[a \leq X \leq b] = \int_a^b f_X(x) \, dx$
 - $E(X) := \int_{-\infty}^{\infty} x f(x) \, dx$
 - $\text{Var}(X) := \int_{-\infty}^{\infty} [x - E(X)]^2 f(x) \, dx$
 - if $F_X$ is the cumulative distribution function(CDF) of $X$,  
$F_X(x) = \int_{-\infty}^x f_X(u) \, du,$
and (if $f_X$ is continuous at $x$) $f_X(x) = \frac{d}{dx} F_X(x)$.
 - the probability density function of a continuous random variable over a single value is zero.  
 $P(X = a) = P(a \leq X \leq a) = \int_a^a f(x) \, dx = 0$

## Expected Value vs Mean vs Average
 - average and mean : mathematically, average and mean are same. So basic formulas used to calculate average and mean are also the same. But the difference between them lies in context in which they are use. The term average is used to estimate an approximate value of a given data in general purpose. However, the use of word in “mean” is specifically used in the context of statistics. In other words, mean is specifically used to represent the average of the statististical data.
(average == mean)
 - mean vs expected value : mean is typically used when we want to calculate the average value of a given sample. This represents the average value of raw data that we’ve already collected. However, expected value is used when we want to calculate the mean of a probability distribution. This represents the average value we expect to occur before collecting any data. 
 In other words, expected value is generalization of the weighted average. Informally, the expected value is the arithmetic mean of the possible values a random variable can take, weighted by the probability of those outcomes. 

### PMF : discrete random variable
A function that gives the probability that a discrete random variable is exactly equal to some value. Sometimes it is also known as the discrete probability density function.  
#### Properties of PMF
- $p : \mathbb{R} \to [0, 1], \ \ p_X(x) = P(X = x) \\$
- $\sum_x p_X(x) = 1, \ \ p_X(x) \geq 0$

### CDF : the probability that the random variable X that takes on a value less than or equal to x
The cumulative distribution function of a real-valued random variable $X$ is the function given by,  
$
F_X(x) = P(X \leq x)
$
#### Properties of CDF
when $f(x)$ is probability density function and $F(x)$ is cumulative distribution function.  

- ${P}(a < X \leq b) = F_X(b) - F_X(a) \\$
- $f(x) = \frac{dF(x)}{dx} \\$
- $F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt \\$ 
- ${P}(X = b) = F_X(b) - \lim_{x \to b^-} F_X(x)$

## Mean Square Error (MSE)
The 'mean squared error (MSE)' or 'mean squared deviation (MSD) of an estimator' (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors.  
The average squared difference between the estimated values and the actual value.  
### Definition
If a vector of $n$ predictions is generated from a sample of $n$ data points on all variables, and $Y$ 
is the vector of observed values of the variable being predicted, with $\hat {Y}$ being the predicted values (e.g. as from a least-squares fit), then the within-sample MSE of the predictor is computed as
$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \hat{Y}_i \right)^2
$  
since the mean is $\left( \frac{1}{n} \sum_{i=1}^{n} \right)$ and error is $\left( Y_i - \hat{Y}_i \right)^2 = \left( e_i \right)^2$
In matrix calculation,   
$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( e_i \right)^2 = \frac{1}{n} \mathbf{e}^T \mathbf{e}
$  
where $e_i$ is $Y_i - \hat{Y}_i$ and $\mathbf{e}$ is $n \times 1$ column vector.