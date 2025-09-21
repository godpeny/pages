# Advertisement Recommendation (ADRec)
## Sequential Recommendation
## Click-Through Rate(CTR) Prediction Model
## Follow The Regularized Leader (FTRL)
https://optimization.cbe.cornell.edu/index.php?title=FTRL_algorithm
https://keras.io/2/api/optimizers/ftrl/

## Radial Basis Function (RBF) kernel
RBF kernel is a popular kernel function used in various kernelized learning algorithms. In particular, it is commonly used in support vector machine classification. Since the value of the RBF kernel decreases with distance and ranges between zero (in the infinite-distance limit) and one (when $x = x'$), it has a ready interpretation as a similarity measure.  
The RBF kernel on two samples $ \mathbf {x} ,\mathbf {x'} \in \mathbb {R} ^{k}$ represented as feature vectors in some input space, is defined as,
$$
{\displaystyle K(\mathbf {x} ,\mathbf {x'} )=\exp \left(-{\frac {\|\mathbf {x} -\mathbf {x'} \|^{2}}{2\sigma ^{2}}}\right)}
$$
Where $ \|\mathbf {x} -\mathbf {x'} \|^{2}$ may be recognized as the squared Euclidean distance between the two feature vectors and $\sigma$ is a free parameter. 

## Stein’s Identity and Kernelized Stein Discrepancy

## Cold Start
Cold start is a potential problem in automated data modelling. It concerns the issue that the system cannot draw any inferences for users or items about which it has not yet gathered sufficient information.

