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
Ensemble means ‘a collection of things’ and in Machine Learning terminology, Ensemble learning refers to the approach of combining multiple ML models to produce a more accurate and robust prediction compared to any individual model.  
The idea is to train a diverse set of weak models on the same modelling task, such that the outputs of each weak learner have poor predictive ability (i.e., high bias), and among all weak learners, the outcome and error values exhibit high variance.  
The set of weak models — which would not produce satisfactory predictive results individually — are combined or averaged to produce a single, high performing, accurate, and low-variance model to fit the task as required.  
The Ensemble learning typically refers to bagging (bootstrap aggregating), boosting or stacking/blending techniques to induce high variance among the base models. We will look over only bagging and boosting.

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

### Bagging (Bootstrapped Aggregation)
![alt text](images/blog24_bagging.png)
 - Bootstrap Sampling: Divides the original training data into ‘N’ subsets and randomly selects a subset with replacement in some rows from other subsets.
 - Base Model Training: For each bootstrapped sample, train a base model independently on that subset of data. These weak models are trained in parallel to increase computational efficiency and reduce time consumption.
 - Prediction Aggregation: To make a prediction on testing data combine the predictions of all base models. For classification tasks, it can include majority voting or weighted majority while for regression, it involves averaging the predictions.
 - Out-of-Bag (OOB) Evaluation: Some samples are excluded from the training subset of particular base models during the bootstrapping method. These “out-of-bag” samples can be used to estimate the model’s performance without the need for cross-validation.
 - Final Prediction: After aggregating the predictions from all the base models, Bagging produces a final prediction for each instance.
#### Bootstrap
![alt text](images/blog24_bagging_bootstrap.png)  
Bootstrap aggregation (bagging) involves training an ensemble on bootstrapped data sets. A bootstrapped set is created by selecting from original training data set with replacement. Thus, a bootstrap set may contain a given example zero, one, or multiple times. Ensemble members can also have limits on the features (e.g., nodes of a decision tree), to encourage exploring of diverse features.

#### Aggregating
#### Random Forest

### Boosting
#### Adaboost

#### Bagging vs Boosting
![alt text](images/blog24_bagging_vs_boosting.png)
배깅은 병렬로 학습하는 반면, 부스팅은 순차적으로 학습합니다. 한번 학습이 끝난 후 결과에 따라 가중치를 부여합니다. 그렇게 부여된 가중치가 다음 모델의 결과 예측에 영향을 줍니다.

오답에 대해서는 높은 가중치를 부여하고, 정답에 대해서는 낮은 가중치를 부여합니다. 따라서 오답을 정답으로 맞추기 위해 오답에 더 집중할 수 있게 되는 것입니다. 

부스팅은 배깅에 비해 error가 적습니다. 즉, 성능이 좋습니다. 하지만 속도가 느리고 오버 피팅이 될 가능성이 있습니다
