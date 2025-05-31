# Machine Learning Strategy
## Background
Why would we need strategy for machine learning? Let me give you an example. Let's say that you get $90$ % accuracy and itis not enough, figure out which of all of these ideas (or other ideas) are worth pursuing and which ones you can safely discard.

 - Collect more data
 - Collect more diverse training set
 - Train algorithm longer with gradient descent
 - Try Adam instead of gradient descent
 - Try bigger network
 - Try smaller network
 - Try dropout
 - Add L-2 regularization
 - Network architecture
   - Activation functions
   - Number of hidden units

## Orthogonalization
Orthogonalization is a system design property that ensures that modification of an instruction or an algorithm component does not create or propagate side effects to other system components.
### Benefits of Orthogonalization
1. Orthogonalization can lead to improved model performance by reducing the complexity and ensuring that each component of the model works efficiently.  
2. Separating concerns through orthogonalization simplifies the debugging process, as issues in one component are less likely to affect others, making the model easier to maintain and update.

For example, when you are driving a car, handle should only control the steering and gear should only control the acceleration. If handle is controlling both steering and acceleration, or vice versa, driving would become much more complex job.

### Chain of Assumptions in Machine Learning
1. Fit training set well on cost function
 - bigger network, Adam Optimizer, ...
2. Fit dev set well on cost function
 - regularization, bigger training sets, ...
3. Fit test set well on cost function
 - bigger dev sets
4. Performs well in real world
 - change dev sets or cost function

Since the sets of method of each criteria are following Orthogonalization, you can use sets of method to tune the algorithm to fit well on each(train/dev/test) sets.  
For example, you can apply regularization or bigger train sets when the model does well on the training sets but poorly on the dev sets. Similarly, you can find the bigger dev sets when your model do well on the dev sets but poorly on the test sets, because your model is overtuned to dev sets. Lastly, you have to fix the dev/test sets distribution or cost function when your model does well on the test sets but poorly on the real world.  
Additionally early stopping is not recommended idea because it can work on multiple criteria simultaneously.


## Single Number Evaluation Metric
If you are building a machine learning system, you will have to measure your algorithm performance. A single number evaluation metric would give a helpful indication of algorithm performance.

Let's see an example.
$$
\begin{array}{|c|c|c|c|}
\hline
\textbf{Classifier} & \textbf{Precision} & \textbf{Recall} & \textbf{F\_1\ Score} \\ \hline
A & 95\% & 90\% & 92.4\% \\ \hline
B & 98\% & 85\% & 91.0\% \\ \hline
\end{array}
$$

As you can see from above example, precision and recall are  evaluation metrics. However, two number evaluation metric is somehow confusing when it comes to choosing either of classifier, because you can't tell which one is better in this case. In this case, you might take the mean of two numbers or F1-score, which is helpful to combine into a single number evaluation.

### Precision and Recall
![alt text](images/blog30_precision_and_recall.png)

Consider a computer program for recognizing dogs (the relevant element) in a digital photograph. Upon processing a picture which contains $10$ cats and $12$ dogs, the program identifies $8$ dogs.  
Of the $8$ elements identified as dogs, only $5$ actually are dogs (true positives), while the other $3$ are cats (false positives).  
$7$ dogs were missed (false negatives), and $7$ cats were correctly excluded (true negatives). The program's precision is then $\frac{5}{8}$ (true positives / selected elements) while its recall is $\frac{5}{12}$ (true positives / relevant elements).

### F-Score

$$
{\displaystyle F_{1}={\frac {2}{\mathrm {recall} ^{-1}+\mathrm {precision} ^{-1}}}=2{\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}={\frac {2\mathrm {TP} }{2\mathrm {TP} +\mathrm {FP} +\mathrm {FN} }}}
$$

### Satisficing and Optimizing Metrics
There are different metrics for assessing a classifier‘s performance, they are called evaluation matrices. They can be categorized as matrices that satisfy and optimize.  
First, metric for optimizing is to maximize as well as possible. While, metric for satisfying  has to be just good enough.
For example, optimizing metric is "Accuracy" and satisfying metric is "Running Time" in the case below.
$$
\begin{aligned}
\text{maximize}\quad & \text{Accuracy} \\[4pt]
\text{subject to}\quad & \text{Running Time} \le 100\ \text{ms}
\end{aligned}
$$

Generally, when there are $N$ metrics, optimizing $1$ metric and satisfying $N-1$ metrics.

## Train, Dev and Test Sets 
Basically you train on the training set, evaluate results on the dev set, and test on the test set.


### Distributions
### Size of Sets
### When to change

## Human Level Performance

### Why compare to Human Level Performance?

### Avoidable Bias
### Bias/Variance with Huaman Level Performance

## Improving Model Performance



