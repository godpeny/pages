# Information Theory
Information theory is the mathematical study of the quantification, storage, and communication of information.
A key measure in information theory is entropy.  

Information theory is based on probability theory and statistics, where quantified information is usually described in terms of bits.
One of the most important measures is called entropy, which forms the building block of many other measures.  
Entropy allows quantification of measure of information in a single random variable.
Another useful concept is mutual information defined on two random variables, 
which describes the measure of information in common between those variables, which can be used to describe their correlation. 

The choice of logarithmic base in the following formulas determines the unit of information entropy that is used.  
A common unit of information is the bit(or shannon), based on the binary logarithm($\log_{2}$). 
https://en.wikipedia.org/wiki/Cross-entropy
Note that an expression of the form $p\log p$ is considered by convention to be equal to zero whenever $p = 0$.  
This is justified because ${\displaystyle \lim _{p\rightarrow 0+}p\log p=0}$ or any logarithmic base.

## Information Content(Self-Information, Shannon information)
It is a basic quantity derived from the probability of a particular event occurring from a random variable. It can be interpreted as quantifying the level of "surprise" of a particular outcome.   
As it is such a basic quantity, it also appears in several other settings, such as "the length of a message needed to transmit the event" given an optimal source coding of the random variable.

It is closely related to entropy, which is the expected value of Information Content of a random variable, quantifying how surprising the random variable is "on average".  
In other words, entropy is the average amount of self-information an observer would expect to gain about a random variable when measuring it.

## Bit(=Shannon)
The shannon is sort of unit of information named after Claude Shannon, the founder of information theory.
IEC 80000-13 defines the shannon as the information content associated with an event when the probability of the event occurring is ⁠$\frac{1}{2}$. 

![alt text](images/blog25_information_theory_bit.png)
If you have one bit, you can specify one of two possibilities, usually written 0 and 1.
![alt text](images/blog25_information_theory_bit2.png)
Same thing applying to when $n$ outcomes. In general, if you have $b$ bits, you can indicate one of $2^{b} = n$ values.  
For example, suppose you want to store a letter of the alphabet. There are $26$ letters, so how many bits do you need? With $4$ bits, you can specify one of $16$ values, so that’s not enough. With $5$ bits, you can specify up to $32$ values, so that’s enough for all the letters, with a few values left over.  
Let's see another example. If flip a coin and tell you the outcome. I have given you one bit of information. If I roll a six-sided die and tell you the outcome, I have given you  $\log_2{6}$ bits of information.

In general, if the number of outcome is $N$, then the outcome contains  $\log_2{𝑁}$ bits of information.

### Relationship With Self-Information
The Bit is unit of self-information. If you measure self‐information using a base‐$2$ logarithm, then the unit is the “bit.”   
Recall that if the number of outcome is $N$, then the outcome contains $\log_2{𝑁}$ bits of information.  
Equivalently, if the probability of the outcome is $p = \frac{1}{N}$, then the information content is $\log_2{\frac{1}{p}} = - \log_2{p} ( = \log_2{N} )$.  
This quantity is called the self-information of the outcome. It measures how surprising the outcome is, which is why it is also called surprisal. 

For example, if your horse has only one chance in $16$ of winning, and he wins, you get $4$ bits of information (along with the payout). But if the favorite wins $75$% of the time, the news of the win contains only $0.42$ bits.

## Axioms of Information Content
Claude Shannon's definition of self-information has to meet below axioms.

1. An event with probability 100% is perfectly unsurprising and yields no information.
2. The less probable an event is, the more surprising it is and the more information it yields.
3. If two independent events are measured separately, the total amount of information is the sum of the self-informations of the individual events.

## Definition of Information Content
Given a real number $b > 1$ and an event $x$ with probability $P$, the information content is defined as follows:
$$
{\displaystyle \mathrm {I} (x):=-\log _{b}{\left[\Pr {\left(x\right)}\right]}=-\log _{b}{\left(P\right)}.}
$$
Above is a unique function of probability that meets the three axioms, up to a multiplicative scaling factor. 
Note that the base $b$ corresponds to the scaling factor above and different choices of base $b$ correspond to different units of information.  
(e.g., when $b = 2$, the unit is the shannon.)

## Entorpy
Entorpy is the expected information content of measurement of $X$.  
It  measures the expected amount of information earned by identifying the outcome of a random trial.  
In other words, Entropy implies that rolling a die has higher entropy than tossing a coin because each outcome of a die toss has smaller probability $\frac{1}{6}$ than each outcome of a coin toss $\frac{1}{2}$.

### Relationship of Entropy and Information Content
Entropy of the random variable $X$ can be defined as below.
$$
{\displaystyle {\begin{alignedat}{2}\mathrm {H} (X)&=\sum _{x}{-p_{X}{\left(x\right)}\log {p_{X}{\left(x\right)}}}\\&=\sum _{x}{p_{X}{\left(x\right)}\operatorname {I} _{X}(x)}\\&{\overset {\underset {\mathrm {def} }{}}{=}}\ \operatorname {E} {\left[\operatorname {I} _{X}(X)\right]},\end{alignedat}}}
$$
It is equal to the expected information content of measurement of $X$.

## Cross Entropy
The cross-entropy between two probability distributions $p$ and $q$, over the same underlying set of events, measures the average number of bits needed to identify an event drawn from the set when the coding scheme used for the set is optimized for an estimated probability distribution $q$, rather than the true distribution $p$.

The cross-entropy of the distribution $q$ relative to a distribution $p$ over a given set is defined as follows:
$$
{\displaystyle H(p,q)=-\operatorname {E} _{p}[\log q],}
$$
Where ${E} _{p}$ is expected value respect to the distribution $p$.

The definition may be formulated using the Kullback–Leibler divergence 
${\displaystyle D_{\mathrm {KL} }(p\parallel q)}$, which represents the divergence of $p$ from $q$ (also known as the relative entropy of 
$p$ with respect to $q$).

$$
{\displaystyle H(p,q)=H(p)+D_{\mathrm {KL} }(p\parallel q)}
$$
Where $H(p)$ is the entropy of $p$.

For discrete probability distributions $p$ and $q$, 
$$
{\displaystyle H(p,q)=-\sum _{x\in {\mathcal {X}}}p(x)\,\log q(x)} 
$$

Similarly, for continuous distributions,
$$
{\displaystyle H(p,q)=-\int _{\mathcal {X}}P(x)\,\log Q(x)\,\mathrm {d} x.} 
$$
Where $P$ and $Q$ be probability density functions of $p$ and $q$.

### Cross Entropy Loss vs Negative Log Likelihood
Cross entropy is negative log likelihood.  
It is because minimizing cross entropy is same as maximizing  log likelihood.

### Cross Entropy Loss
https://www.geeksforgeeks.org/what-is-cross-entropy-loss-function/
https://stackoverflow.com/questions/41990250/what-is-cross-entropy
https://jmlb.github.io/ml/2017/12/26/Calculate_Gradient_Softmax/


### Gradient of Cross Entropy Loss

#### Gradient of Cross Entropy Loss w.r.t. input of softmax 
$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{a} \odot \left( \mathbf{g} - \mathbf{a}^\top \mathbf{g} \right)
$$

https://hyunw.kim/blog/2017/10/27/KL_divergence.html
https://en.wikipedia.org/wiki/Cross-entropy