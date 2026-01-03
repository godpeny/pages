# Clustering
Clustering is an unsupervised machine learning technique designed to group unlabeled examples based on their similarity to each other. 
If the examples are labeled, this kind of grouping is called "classification".

<img src="images/blog38_clustering.png" alt="Clustering" width="600"/>   

As you can see  the data forms three clusters, even without a formal definition of similarity between data points. In real-world applications, however, you need to explicitly define a similarity measure, or the metric used to compare samples, in terms of the dataset's features. As the number of features increases, combining and comparing features becomes less intuitive and more complex.

After clustering, each group is assigned a unique label called a cluster ID. Clustering is powerful because it can simplify large, complex datasets with many features to a single cluster ID.

## Clustering Algorithms
### Centroid-based clustering
The centroid of a cluster is the arithmetic mean of all the points in the cluster. 

Centroid-based clustering organizes the data into non-hierarchical clusters. Centroid-based clustering algorithms are efficient but sensitive to initial conditions and outliers. Of these, <b>k-means</b> is the most widely used. It requires users to define the number of centroids, k, and works well with clusters of roughly equal size.

### Density-based Clustering
Density-based clustering connects contiguous areas of high example density into clusters. This allows for the discovery of any number of clusters of any shape. Outliers are not assigned to clusters. These algorithms have difficulty with clusters of different density and data with high dimensions.

### Distribution-based Clustering
This clustering approach assumes data is composed of probabilistic distributions, such as Gaussian distributions. As distance from the distribution's center increases, the probability that a point belongs to the distribution decreases. 

### Hierarchical Clustering
Hierarchical clustering creates a tree of clusters. Hierarchical clustering, not surprisingly, is well suited to hierarchical data, such as taxonomies.

## Clustering Workflows
## Data preparation

## K-Means Clustering
## Supervised Deep Neural Network for Clustering

## References
- https://developers.google.com/machine-learning/clustering
