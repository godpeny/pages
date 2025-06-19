# Softmax and Cross Entropy
## Softmax
### Basics

### How Softmax works?

### Gradient of Softmax
https://medium.com/@jsilvawasd/softmax-and-backpropagation-625c0c1f8241

https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

### Loss Function of Softmax

## Cross Entropy
https://en.wikipedia.org/wiki/Cross-entropy

### Cross Entropy Loss vs Negative Log Likelihood
Cross entropy is negative log likelihood.  
It is because minimizing cross entropy is same as maximizing  log likelihood.

### Cross Entropy Loss
https://www.geeksforgeeks.org/what-is-cross-entropy-loss-function/
https://stackoverflow.com/questions/41990250/what-is-cross-entropy


### Gradient of Cross Entropy Loss

#### Gradient of Cross Entropy Loss w.r.t. input of softmax 
$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{a} \odot \left( \mathbf{g} - \mathbf{a}^\top \mathbf{g} \right)
$$
