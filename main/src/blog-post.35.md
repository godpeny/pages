# Contrasive Learning
Contrastive Learning is a Machine Learning paradigm where unlabeled data points are juxtaposed against each other to teach a model which points are similar and which are different. That is, as the name suggests, samples are contrasted against each other, and those belonging to the same distribution are pushed towards each other in the embedding space. In contrast, those belonging to different distributions are pulled against each other.

The basic contrastive learning framework consists of selecting a data sample, called “anchor,” a data point belonging to the same distribution as the anchor, called the “positive” sample, and another data point belonging to a different distribution called the “negative” sample. Then, the Self-Supervised Learning(SSL) model tries to minimize the distance between the anchor and positive samples, i.e., the samples belonging to the same distribution, in the latent space, and at the same time maximize the distance between the anchor and the negative samples.

For example,
<img src="images/blog35_contrasive_learning.png" alt="Contrasive Learning" width="400"/>   

As shown in the example above, two images belonging to the same class lie close to each other in the embedding space (“d+”), and those belonging to different classes lie at a greater distance from each other (“d-”). Thus, a contrastive learning model (denotes by “theta” in the example above) tries to minimize the distance “d+” and maximize the distance .

## Self-Supervised Learning(SSL)
Self-supervised learning (SSL) is a paradigm in machine learning where a model is trained on a task using the data itself to generate supervisory signals, rather than relying on externally-provided labels. 

## Loss
### Contrasive Loss
Contrastive loss takes the output of the network for a positive example and calculates its distance to an example of the same class and contrasts that with the distance to negative examples. Said another way, the loss is low if positive samples are encoded to similar (closer) representations and negative examples are encoded to different (farther) representations.
$$
\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \ne i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
$$
- 분자: positive pair의 similarity
- 분모: 자기 자신을 제외한 모든 similarity (positive + negative)

즉, $ - \log{\frac{p}{p+n}}$ 구조에서 $p,n$ 은 모두 지수함수 이기에 최소가 되려면 log 안이 1이 되어야 한다. 즉, 분자(p)가 커지거나 분모의 n이 작아지는 방향으로 학습되는 것이다.

### Triplet Loss
The loss function is defined using triplets of training points of the form (A,P,N). In each triplet, A denotes a reference point of a particular identity, P(called a "positive point") denotes another point of the same identity in point A, and N (called a "negative point") denotes a point of an identity different from the identity in point A and P.

Let $x$ be some point and let $f(x)$ be the embedding of $x$ in the finite-dimensional Euclidean space. We assemble $m$ triplets of points from the training dataset. The goal of training here is to ensure that, after learning, the following condition (called the "triplet constraint") is satisfied by all triplets $(A^{(i)},P^{(i)},N^{(i)})$ in the training data set.
$$
\Vert f(A^{(i)})-f(P^{(i)})\Vert _{2}^{2}+\alpha <\Vert f(A^{(i)})-f(N^{(i)})\Vert _{2}^{2}
$$
The variable $\alpha$ is a hyperparameter called the margin, and its value must be set manually. In the FaceNet system, its value was set as 0.2.

Thus, the full form of the function to be minimized is the following.
$$
L=\sum _{i=1}^{m}\max {\Big (}\Vert f(A^{(i)})-f(P^{(i)})\Vert _{2}^{2}-\Vert f(A^{(i)})-f(N^{(i)})\Vert _{2}^{2}+\alpha ,0{\Big )}
$$

Triplet loss innovates by considering relative distances. Its goal is that the embedding of an anchor (query) point be closer to positive points than to negative points (also accounting for the margin). It does not try to further optimize the distances once this requirement is met. This is approximated by simultaneously considering two pairs (anchor-positive and anchor-negative), rather than each pair in isolation.

### InfoNCE Loss
Contrastive Learning (CL) is an Self-Supervised Learning(SSL) paradigm where an encoder $f : X → Z$ learns to map observations x to latent vectors z.

InfoNCE는 대조 학습(CL)에서 가장 널리 사용되는 손실 함수입니다. 인코더 f가 입력 x를 잠재 벡터(latent vector) z로 매핑할 때 비슷한 샘플(Positive pair) 간의 거리는 좁히고, 다른 샘플(Negative pair) 간의 거리는 멀어지게 만드는 역할을 합니다.   

$$
L_{\text{INCE}}= \mathbb{E}_{\substack{\mathbf{x},\mathbf{x}^+ \\ \lbrace\mathbf{x}_i^-\rbrace}} \left[ -\ln \frac{e^{\mathbf{f}^\top(\mathbf{x})\mathbf{f}(\mathbf{x}^+)/\tau}}{e^{\mathbf{f}^\top(\mathbf{x})\mathbf{f}(\mathbf{x}^+)/\tau} + \sum_{i=1}^M e^{\mathbf{f}^\top(\mathbf{x})\mathbf{f}(\mathbf{x}_i^-)/\tau}} \right],
$$
- $x$ (Anchor): 기준이 되는 입력 데이터입니다.
- $x^{+}$ (Positive): $x$ 와 유사한 데이터(예: 같은 이미지의 증강된 버전)로, 조건부 분포 $p(x^{+}| x)$에서 샘플링됩니다.
- $x_i^{-}$ (Negatives): $x$ 와 다른 데이터(예: 데이터셋에서 무작위 추출)로, 분포 $p(x)$에서 독립적(i.i.d.)으로 추출된 $M$개의 샘플입니다.
- $\mathbf{x},\mathbf{x}^+$: 두 벡터 간의 유사도(내적)를 나타냅니다.
- $\tau$ (Temperature): 분포의 스케일을 조절하는 스칼라 값입니다.

### Reference
- https://web3.arxiv.org/pdf/2511.23312

### Logistic Loss
The log loss is only defined for two or more labels.(2개: sigmoid, 3개 이상: softmax)  
$$
\ell_{n=2} = -\left( y \log p + (1 - y)\log(1 - p) \right) \\
\ell_{n>2} = -\sum_{i=1}^{K} y_i \log p_i
$$

## Reference
https://arxiv.org/pdf/2002.05709
https://www.v7labs.com/blog/contrastive-learning-guide