# Generative Learning Algorithms (GLA)
## Basics (Discriminative Models vs Generative Models)
### Generative Learning Models (GLA)
 - Generative model is a model of the conditional probability of the observable $X$, given a target $y$, symbolically, $P(Y \mid X=x)$
 - Generative model is a statistical model of the joint probability distribution $P(X, Y)$ on a given observable variable $X$ and target variable $Y$. A generative model can be used to "generate" random instances (outcomes) of an observation $x$.
 - Generative model includes the distribution of the data itself, and tells you how likely a given example is. e.g., models that predict the next word in a sequence are typically generative models (usually much simpler than GANs) because they can assign a probability to a sequence of words.
 - For example, Naive Bayes classifier, GAN.
### Discriminative Learning Models (DLA)
 - Discriminative model is a model of the conditional probability of the target $Y$, given an observation $x$, symbolically, $P(Y \mid X=x)$.
 - Discriminative model is a model of the conditional probability $P(Y\mid X=x)$ of the target $Y$, given an observation $x$. It can be used to "discriminate" the value of the target variable $Y$, given an observation $x$.
 - Discriminative model ignores the question of whether a given instance is likely, and just tells you how likely a label is to apply to the instance.
 - For example, logistic regression.


## Gaussian Discriminant Analysis (GDA)

### Multivariate Gaussian Distribution

### GDA vs Logistic Regression

## Naive Bayes

### Laplace Smoothing

### Multi-Variate Bernoulli Event Model vs Multinomial Event Model