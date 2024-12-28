# Deision Tree and Ensemble
## Preliminaries
### Covariance
Covariance is a measure of the joint variability of two random variables. It is the sign of the covariance, therefore, shows the tendency in the linear relationship between the variables.  
For example, if the covariance is positive, greater values of one variable mainly correspond with greater values of the other variable, and the same holds for lesser values (that is, the variables tend to show similar behavior).  
If the covarinace is negative, the greater values of one variable mainly correspond to lesser values of the other (that is, the variables tend to show opposite behavior)
$$
\text{cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$
Also, it can be wrtitten as,
$$
\text{cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] \\
= \mathbb{E}[XY - X\mathbb{E}[Y] - \mathbb{E}[X]Y + \mathbb{E}[X]\mathbb{E}[Y]] \\
= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]\mathbb{E}[Y] + \mathbb{E}[X]\mathbb{E}[Y] \\
= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$
### Variance
Variance is the expected value of the squared deviation from the mean of a random variable.
$$
\text{Var}(X) = \mathbb{E}[(X - \mu)^2].
$$
### Relationship Between Variance and Covariance
$$
\text{Cov}(X_i, X_i) = \text{Var}(X_i).
$$
Using the fact above, the sum of variance can be written as
$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \sum_{j=1}^n \text{Cov}(X_i, X_j) = \\ 
\sum_{i=1}^n \text{Var}(X_i) + 2 \sum_{1 \leq i < j \leq n} \text{Cov}(X_i, X_j) = \\
\sum_{i=1}^n \text{Var}(X_i) + \sum_{i \neq j} \text{Cov}(X_i, X_j)
$$

### Correlation
In statistics, correlation or dependence is any statistical relationship, whether causal or not, between two random variables or bivariate data.  
Although in the broadest sense, "correlation" may indicate any type of association, in statistics it usually refers to the degree to which a pair of variables are linearly related.

### Correlation Coefficient
A correlation coefficient is a numerical measure of some type of linear correlation, meaning a statistical relationship between two variables.  
The variables may be two columns of a given data set of observations, often called a sample, or two components of a multivariate random variable with a known distribution.

There are several different measures for the degree of correlation in data, depending on the kind of data(Whether the data is a measurement, ordinal, or categorical). For example, Pearson, Intra-class, Rank and so on.

    
### Pearson Correlation Coefficient
The Pearson Correlation Coefficient is a correlation coefficient that measures linear correlation between two sets of data.  
It is the ratio between the covariance of two variables and the product of their standard deviations.
$$
\rho_{X,Y} = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
$$
Where "cov" is covariance and $\sigma_X$ and $\sigma_Y$ are the standard deviation of $X$ and $Y$. This fomular can also be written as below,
$$
\rho_{X,Y} = \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sigma_X \sigma_Y} \\
$$
Where $\mu_X$ and $\mu_Y$ are the mean of $X$ and $Y$.
Above is also can be written as,
$$
\rho_{X, Y} = \frac{\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]}{\sqrt{\mathbb{E}[X^2] - (\mathbb{E}[X])^2} \sqrt{\mathbb{E}[Y^2] - (\mathbb{E}[Y])^2}}.
$$
It is because, 
$$
\mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mathbb{E}[X\mu_Y] - \mathbb{E}[\mu_X Y] + \mathbb{E}[\mu_X \mu_Y].
$$
While $ \mathbb{E}[X\mu_Y] =  \mathbb{E}[X\mu_Y]  = \mathbb{E}[\mu_X \mu_Y]$.

### Covariance and Correlation
In probability theory and statistics, the mathematical concepts of covariance and correlation are very similar. Since both describe the degree to which two random variables or sets of random variables tend to deviate from their expected values in similar ways.

$$
\textbf{covariance} \\
\text{cov}_{XY} = \sigma_{XY} = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] \\

\textbf{correlation} \\
\text{corr}_{XY} = \rho_{XY} = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] / (\sigma_X \sigma_Y), \\

\text{so that} \\

\rho_{XY} = \sigma_{XY} / (\sigma_X \sigma_Y)
$$

## Decision Tree
### Cross-Entropy
## Ensmeble
### Basic Probability Theory in Ensemble
Since $\text{Var}(X_i) = \sigma^2$ and $\text{Var}(aX) = a^2 \text{Var}(X)$, variance of mean of $X$($\bar{X})$ is,
$$
\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n} \sum_i X_i \right) = \frac{\sigma^2}{n}
$$
When variables are only identically distributed, using above derivation we can find out that,
$$
\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n} \sum_i X_i \right) \text{(1)} \\
= \frac{1}{n^2} \sum_{i,j} \text{Cov}(X_i, X_j) \text{(2)} \\
= \frac{n\sigma^2}{n^2} + \frac{n(n-1)\rho\sigma^2}{n^2} \text{(3)} \\
= \rho\sigma^2 + \frac{1-\rho}{n} \sigma^2 \text{(4)}
$$
(2) and (3) is derived from the relationship between variance and covariance, which is,
$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \sum_{j=1}^n \text{Cov}(X_i, X_j)
 = \sum_{i=1}^n \text{Var}(X_i) + \sum_{i \neq j} \text{Cov}(X_i, X_j)
$$
Also in (3), we know that varinace of the mean is $\frac{\sigma^2}{n}$ and any sets of $X_i$ and $X_j$  are correlated by a factor $\rho$ when $i \neq j$.  
Therefore below can be derived as shown in (3).
$$
\text{Since } \text{Var}(X_i)= \sigma^2, \\ 
\sum_{i=1}^n \text{Var}(X_i) = n \sigma^2
$$
$$
\sum_{i \neq j} \text{Cov}(X_i, X_j) = (n^2-n)\rho\sigma^2
$$
Multiplying two with $\frac{1}{n^2}$ and combined together, we can understsand (4).

### Bagging
### Boosting

