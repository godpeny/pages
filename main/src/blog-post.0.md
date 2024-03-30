# Stub

## Terminology
### Variables vs Parameters
- Variables : values that are input to a model.
  - i.e. input features in a neural network. ( = x)
- Parameters : values that are learned by the model during training.
  - i.e. weights, biases in a neural network. ( = W, b) 
  - theta : set of parameters in a model.

##### Train, Dev, Test Set
train on the training set, evaluate results on the dev set, and test on the test set.
so do not test your model on the test set until you have finished handling overfitting.

## Math
### Statistics
#### L2 Norm
#### Gaussian Distribution (= Normal Distribution)
#### Probability vs Likelihood
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


#### Cumulative Distribution Function(CDF) vs Probability Density Function(PDF) vs Probability Mass Function(PMF)
#### Expectation vs Mean vs Average

### Calculus
#### Chain Rule
#### Jacobian vs Gradients vs Hessian
#### Differential vs Derivative vs Gradient
#### Epsilon in Calculus
- Epsilon is a small positive number, often used in numerical computation to avoid division by zero or taking the logarithm of zero.
#### Epsilon-Delta Definition of Limit


### Linear Algebra
#### Convex and Concave Function
Convex function : function is convex if the line segment between any two distinct points on the graph of the function lies above the graph between the two points. (볼록 함수 - 아래로 볼록)
Concave function : function is concave if the line segment between any two distinct points on the graph of the function lies below the graph between the two points. (오목 함수 - 아래로 오목)
#### Vector and Matrix
##### Row Vector vs Column Vector
##### Span, Range, Rank and Null Space
##### Vector and Matrix Multiplication
