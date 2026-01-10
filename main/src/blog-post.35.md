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
### Triplet Loss
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

## Reference
https://arxiv.org/pdf/2002.05709
https://www.v7labs.com/blog/contrastive-learning-guide