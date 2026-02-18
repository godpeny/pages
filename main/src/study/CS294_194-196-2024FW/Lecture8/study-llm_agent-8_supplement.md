# Ring vs Semi-Ring
- Ring: “음수까지 있는 완전한 대수 구조”
- Semi-Ring: “음수가 없는 구조”

Ring = Semiring + Additive Inverses

## Ring
A ring is a set R equipped with two binary operations + (addition) and ⋅ (multiplication) satisfying the following three sets of axioms, called the ring axioms.

1. R is an abelian group under addition, meaning that,
$(a + b) + c = a + (b + c)$ for all a, b, c in R (that is, + is associative).  
$a + b = b + a$ for all a, b in R (that is, + is commutative).
There is an element 0 in R such that a + 0 = a for all a in R (that is, 0 is an additive identity).  
For each a in R there exists −a in R such that a + (−a) = 0 (that is, −a is the additive inverse of a).

2. R is a monoid under multiplication, meaning that,
(a · b) · c = a · (b · c) for all a, b, c in R (that is, ⋅ is associative).

There is an element 1 in R such that a · 1 = a and 1 · a = a for all a in R (that is, 1 is a multiplicative identity).
3. Multiplication is distributive with respect to addition, meaning that:
a · (b + c) = (a · b) + (a · c) for all a, b, c in R (left distributivity).
(b + c) · a = (b · a) + (c · a) for all a, b, c in R (right distributivity).

## Semi-Ring
Ring without additive inverse (For each a in R there exists −a in R such that a + (−a) = 0 (that is, −a is the additive inverse of a).

# Extreme Point Theorem
$$
\max \; c^\top x
\quad \text{s.t. } x \in \Omega \\[5pt]
\Omega = \{\, x \in \mathbb{R}^n \mid Ax \le b,\; x \ge 0 \,\}
$$
$\Omega$ (Feasible Region) 가 비어 있지 않고(bound되지 않았더라도) 최적해가 존재한다면, 적어도 하나의 최적해는 $\Omega$ 의 극점(extreme point, 꼭짓점) 에서 달성된다.

# Group
A group is a set $G$ together with a binary operation on ⁠$G$⁠, here denoted "$\cdot$, that combines any two elements $a$ and $b$ of $G$ to form an element of ⁠$G$⁠, denoted ⁠$ a\cdot b$ such that the following three requirements, known as group axioms, are satisfied.

- Associativity: For all ⁠$a,b,c$ in $G$, one has ⁠$(a\cdot b)\cdot c=a\cdot (b\cdot c)$
- Identity element: There exists an element $e$ in $G$ such that, for every $a$ in ⁠$G$⁠, one has ⁠$e\cdot a=a$⁠ and ⁠$a\cdot e=a$.  
Such an element $e$ is unique. It is called the identity element of the group.
- Inverse element: For each $a$ in ⁠$G$⁠, there exists an element $b$ in $G$ such that $a\cdot b=e$ and ⁠$b\cdot a=e$ where $e$ is the identity element.  
For each ⁠$a$⁠, the element $b$ is unique; it is called the inverse of $a$ and is commonly denoted ⁠$a^{-1}$.

# Abelian group
The group operation is commutative. (Associativity, Identity Element,Inverse Element + Commutativity )

- Commutativity
For all $a$, $b$ in $G$, $a\cdot b=b\cdot a$

# Ring homomorphism
In mathematics, a ring homomorphism is a structure-preserving function between two rings. More explicitly, if R and S are rings, then a ring homomorphism is a function f : R → S that preserves addition, multiplication and multiplicative identity.
$$
\begin{aligned}f(a+b)&=f(a)+f(b),\\f(ab)&=f(a)f(b),\\f(1)&=1,\end{aligned}
$$
for all a, b in R.

# Mixed Integer Programming
Mixed integer programming (MIP) is a mathematical optimization technique that solves problems involving a mix of continuous variables (which can have any value, including decimals and fractions), discrete variables (which must be countable whole numbers), and binary variables (which can only take values 0 or 1).

<img src="./mip-model-and-solution-space.svg" alt="Mixed Integer Programming" width="400"/>    
Abvoe is an example of a mixed integer programming (MIP) model and its solution space: $x$ and $y$ are the decision variables, and $z$ is the objective function. 
The inequalities form the constraint boundaries, represented as lines. Blue dots indicate feasible (valid) solutions that satisfy all constraints, while the green dot marks an optimal solution that maximizes the objective.

# Fourier Transform
The Fourier transform (FT) is an integral transform that takes a function as input, and outputs another function that describes the extent to which various frequencies are present in the original function. The output of the transform is a complex valued function of frequency.
$$
{\displaystyle {\widehat {f}}(\xi )=\int _{-\infty }^{\infty }f(x)\ e^{-i2\pi \xi x}\,dx,\quad \forall \xi \in \mathbb {R} .}    
$$


https://en.wikipedia.org/wiki/Fourier_transform