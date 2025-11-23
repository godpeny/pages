# Contrasive Learning
Contrastive Learning is a Machine Learning paradigm where unlabeled data points are juxtaposed against each other to teach a model which points are similar and which are different. That is, as the name suggests, samples are contrasted against each other, and those belonging to the same distribution are pushed towards each other in the embedding space. In contrast, those belonging to different distributions are pulled against each other.

The basic contrastive learning framework consists of selecting a data sample, called “anchor,” a data point belonging to the same distribution as the anchor, called the “positive” sample, and another data point belonging to a different distribution called the “negative” sample. Then, the Self-Supervised Learning(SSL) model tries to minimize the distance between the anchor and positive samples, i.e., the samples belonging to the same distribution, in the latent space, and at the same time maximize the distance between the anchor and the negative samples.

For example,
<img src="images/blog35_contrasive_learning.png" alt="Contrasive Learning" width="400"/>   

As shown in the example above, two images belonging to the same class lie close to each other in the embedding space (“d+”), and those belonging to different classes lie at a greater distance from each other (“d-”). Thus, a contrastive learning model (denotes by “theta” in the example above) tries to minimize the distance “d+” and maximize the distance .

## Loss
### Contrasive Loss
### Triplet Loss
### InfoNCE Loss
### Logistic Loss

## Reference
https://arxiv.org/pdf/2002.05709
https://www.v7labs.com/blog/contrastive-learning-guide