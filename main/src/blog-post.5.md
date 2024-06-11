# Statistics
## L2 Norm
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
 - E(X) = = mu(mean) = integral(x * f(x) dx) 
 - Var(X) = integral((x - E(X))^2 * f(x) dx)  
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
