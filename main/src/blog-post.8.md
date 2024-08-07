# Generalized Linear Models (GLM)
## Basics
in statistics, a generalized linear model is a flexible generalization of ordinary linear regression. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.  
in a generalized linear model, each outcome $y$ of the dependent variables is assumed to be generated from a particular distribution in an exponential family, a large class of probability distributions that includes the normal, binomial, Poisson and gamma distributions, among others.  

## Exponential Family
an exponential family is a parametric set of probability distributions of a certain form, specified below.  
this special form is chosen formathematical convenience, includingthe enabling of the user to calculateexpectations, covariances usingdifferentiation based on some usefulalgebraic properties, as well as forgenerality, as exponential familiesare in a sense very natural sets ofdistributions to consider. 
$
{\displaystyle \ f_{X}\!\left(x\ {\big |}\ \theta \right)=h(x)\ \exp {\bigl [}\ \eta (\theta )\cdot T(x)-A(\theta )\ {\bigr ]}\ }
$

- sufficient statistic $T(x)$
- naturla parameter $\eta$
- log-partition function $A(\eta)$ : logarithm of a normalization factor, without which ${\displaystyle \ f_{X}\!\left(x\ {\big |}\ \theta \right)}$ would not be a probability distribution:  
$A(\eta) = \log \left( \int_{X} h(x) \exp(\eta(\theta) \cdot T(x)) \, dx \right)$

## 3 Assumptions
GLM consists of three elements below.
1. A particular distribution for modeling $Y$ from among those which are considered exponential families of probability distributions.  
2. A linear predictor $\eta = X\beta$.  
3. A link function $g$ such that $\mathbb{E}(Y \mid X) = \mu = g^{-1}(\eta)$.


## Linear Regression

### Basic

### Margin and Loss Function

### Batch / Stochastic Gradient Descent

### Normal Equation

### Locally Weighted Linear Regression
Rather than learning a fixed set of parameters as is done in ordinary linear regression, parameters $\theta$ are computed individually for each query point $x$.
$
J(\theta) = \sum_{i=1}^{m} w^{(i)} \left( \theta^T x^{(i)} - y^{(i)} \right)^2
$

## Logistic Regression

### Probability Interpretation

### Basics

### Newton's Method

## Softmax Regression