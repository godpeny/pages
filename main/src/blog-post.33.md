# Advertisement Recommendation (ADRec)
## CPA, CPC, CPM, CPP, CPR, CTR, TRP, Reach and Frequency
<b>CPA(Cost per Action) </b>  
CPA는 회원가입, sALES, 홈페이지 방문 등 Action 당 단가를 말한다.  
가장 대표적으로 일컬어지는 것이 회원 가입당 단가, 배너를 클릭해서 광고주 홈페이지에서 회원가입을 할 경우, 배너를 통해 회원가입까지 간 사람들을 1명 당 단가를 책정해서 광고집행금액을 결정하는 방식을 CPA라고 한다.

<b>CPC(Cost per Click)</b>  
CPC는 광고 클릭당 광고비, 광고 1 클릭을 얻기위해 소요되는 광고 비용을 말한다.
동일한 CPM으로 광고를 집행하였더라도, CTR(클릴율)이 높으면, CPC는 떨어지므로 더 효율적인
광고를 집행하였다고 할 수 있다.

<b>CPM (Cost per Mile)(노출 1,000회에 대한 단가)</b>   
CPM은 노출수에 따라 광고비를 내는 정액제 방식으로 CPM은 1000회의 광고 노출에 드는 광고비를 말한다.
CPM이 5000이라면 1회의 노출에 드는 광고비는 5원이라고 할 수 있음. Mile은 로마술자에서 1,000을 나타냄

<b> Reach and Frequency </b>  
Reach의 뜻은 '도달'로서 광고계에서는 풀이하면 '광고나 마케팅 메시지가 이용자에게 알려지거나 전달되는 넓이'를 말한다. 그래서 나같은 경우에는 이 용어를 이해하려고 할 때 '이용자'에 초점을 맞췄다. 그리고 도달률은 전체 이용자 수 대비 메시지가 전달되는 정도를 백분율로 나타낸 것이다.  
Frequency는'빈도'라는 뜻을 가진 영어단어로써, '한 이용자에게 같은 광고나 마케팅 메시지가 보이는 횟수'를 말한다. 즉 '1인당 광고 노출 수'를 의미한다. Frequency에서 주의할 점은 모든 이용자에게 보이는 횟수가 아닌 '한 이용자'라는 점이고 또 '같은' 광고나 마케팅 메시지에 초점을 맞췄다는 것이다.  

<b> TRP (Taget Rationg Point) </b>  
'일반적으로 매체 효과를 평가할 때 사용하는 지수인 GRP를 타겟 대상으로 적용시킨 것'이다. 그래서 공식이 GRP와 많이 닮아 있다. TRP = Tagret Reach X Frequency. 

<b>CPP (Cost per TRP)</b>  
TRP 1포인트를 올리는데 들어간 비용.  

<b>CPR (Cost per Reach) </b>
CPR은 Reach 1%를 올리는데 들어간 비용을 말한다.  

<b>CTR Click-Through Rate</b>  
CTR은 일반적으로 말하는 '클릭율'로서 광고 노출 대비 광고를 클릭한 비율을 백분율로 환산한 수치를 말한다. CTR=100*(Click Impressions)

## Bayesian Personalized Ranking (BPR)
Bayesian Personalized Ranking is a machine learning algorithm specifically designed for enhancing the recommendation process. It operates under a pairwise ranking framework where the goal is not just to predict the items a user might like but to rank them in the order of potential interest. Unlike traditional methods that might predict absolute ratings, BPR focuses on getting the order of recommendations right.

BPR works by maximizing the posterior probability of a user preferring a known positive item over a randomly chosen negative item.

Reference - https://www.geeksforgeeks.org/machine-learning/recommender-system-using-bayesian-personalized-ranking/  
https://arxiv.org/pdf/1205.2618

## Sequential Recommendation
https://arxiv.org/abs/1511.06939
https://arxiv.org/abs/1808.09781
https://arxiv.org/abs/1904.06690

semantic ID?
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
https://arxiv.org/pdf/1608.04471

## Cold Start
Cold start is a potential problem in automated data modelling. It concerns the issue that the system cannot draw any inferences for users or items about which it has not yet gathered sufficient information.

## Calibration in Recommendation
A classification algorithm is called calibrated if the predicted proportions of the various classes agree with the
actual proportions of data points in the available data. Analogously(Similarly), the goal of calibrated recommendations is to reflect the various interests of a user in the recommended list, and with their appropriate proportions.  

Reference - https://dl.acm.org/doi/pdf/10.1145/3240323.3240372

## Models
https://medium.com/@lonslonz/추천-모델-개발-2-딥러닝-모델-29dbf704715
## Wide and Deep Model
https://arxiv.org/pdf/1606.07792

## Deep and Cross Model
https://arxiv.org/abs/1708.05123

## Two Tower Model
https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf
https://storage.googleapis.com/gweb-research2023-media/pubtools/6090.pdf

## Deep Learning Recommendation Mode(DLRM)
https://arxiv.org/pdf/1906.00091

## Tabular Data Modeling Using Contextual Embeddings(TabTransformer)
https://arxiv.org/pdf/2012.06678