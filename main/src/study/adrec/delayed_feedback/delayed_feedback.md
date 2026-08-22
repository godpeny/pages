# Delayed Feedback
### Preliminaries
#### PU Learning
Positive-Unlabeled (PU) Learning 은 양성 샘플(관심 대상 인스턴스)과 라벨이 없는 데이터(클래스 라벨을 알 수 없는 인스턴스)만 존재하는 데이터셋을 다룰 때 사용되는 머신러닝 패러다임의 한 종류입니다. 이러한 시나리오는 사기 탐지, 이상 탐지, 희귀 질환 진단, 감정 분석과 같은 다양한 현실 세계 애플리케이션에서 발생하며, 음성 샘플(양성 클래스에 속하지 않는 인스턴스)을 구하는 것이 까다롭거나 비용이 많이 들거나 혹은 전혀 불가능할 때 주로 사용됩니다.

##### PU Learning의 주요 접근 방식
- 인스턴스 선택 방법: 학습을 보완하기 위해 라벨이 없는 데이터에서 신뢰할 수 있는 음성 인스턴스를 선택하는 것을 목표로 합니다.
- PU-SVM (Positive-Unlabeled Support Vector Machine): 양성 클래스의 가중치를 재조정하고 양성과 라벨 미지정 인스턴스를 구분하는 결정 함수를 도입하여 표준 SVM을 PU Learning 환경에 맞게 변형합니다.
- 신뢰도 추정 기반 PU-Learning: 라벨이 없는 인스턴스가 양성 클래스에 속할 확률을 추정하여 학습 과정에서 보다 정보에 기반한 의사결정을 내릴 수 있도록 합니다.

## Modeling Delayed Feedback in Display Advertising
### Delayed Feedback 문제

광고 노출 혹은 클릭이 발생한 시점부터 실제 사용자가 전환을 일으키기까지는 최대 한 달이라는 긴 시간이 걸릴 수 있습니다. 이처럼 피드백이 늦게 오면 모델을 학습시키는 데 방해가 됩니다. 수집 기간(Matching Window)을 너무 짧게 잡으면 나중에 결국 전환할 사용자를 '전환하지 않은 사용자(Negative)'로 잘못 오분류(Mislabeling)하게 되고, 수집 기간을 너무 길게 잡으면 모델을 학습시킬 데이터가 너무 과거의 것이 되어 변화하는 최신 캠페인 트렌드를 신속하게 반영하지 못하는 정체 현상이 발생합니다.

### 제안: Delayed Feedback Model 도입

논문에서는 이 문제를 해결하기 위해 전환 지연 시간(Conversion Delay)을 직접 캡처하는 추가 확률 모델을 제안합니다.
아직 전환이 발생하지 않은 사용자 데이터를 학습시킬 때 다음과 같이 직관적으로 작동합니다.  
경과 시간 > 예측 지연 시간: 클릭 후 경과시간(Elapsed Time)이 모델이 예측한 지연 시간보다 충분히 길다면, 이 사용자는 앞으로도 전환하지 않을 부정 샘플(Negative Sample)로 취급합니다.  
경과 시간이 너무 짧을 때: 클릭한 지 얼마 지나지 않아 아직 전환 여부를 확신하기 어렵다면, 섣불리 부정 샘플로 분류하지 않고 학습 세트에서 제외(Discard)할 수 있도록 확률적으로 가중치를 조절합니다.  
이를 위해 사용자가 '궁극적으로 전환할 확률'을 예측하는 모델(로지스틱 회귀)과 '전환한다면 그 지연 시간이 얼마나 될지' 예측하는 모델(Survival Analysis 기반 지수 분포)을 결합하여 동시에 학습시킵니다.

### Conversion Rate Prediction
#### eCPM 계산 및 수식 분해
광고주가 전환당 비용(CPA)을 지불하는 모델에서 광고 노출 1회당 기대 가치인 eCPM은 아래 공식으로 계산됩니다.

$$\text{eCPM} = \text{CPA} \times \Pr(\text{click}) \times \Pr(\text{conversion} \mid \text{click})$$

즉, CPA 입찰가에 클릭 확률($\Pr(\text{click})$)과 클릭 대비 전환 확률($\Pr(\text{conversion} \mid \text{click})$)을 모두 곱해 가치를 구합니다.

### MODEL
#### 변수 정의
1. $X$ : 사용자의 데모그래픽, 광고 지면 정보, 과거 행동 이력 등 예측에 사용되는 특성(Features)들의 집합입니다.
2. $Y \in \{0, 1\}$ : 모델 학습 시점에 실제 전환이 관측되었는지 여부를 나타냅니다.  
   $(Y = 1)$: 전환이 발생함.  
   $(Y = 0)$: 아직 전환이 발생하지 않음.
3. $C \in \{0, 1\}$ : 사용자가 궁극적으로(시간이 아무리 오래 걸려도) 전환을 할 것인가에 대한 여부입니다. (실제 환경에서는 숨겨진 변수(Latent Variable)입니다.)
4. $D$ : 사용자가 광고를 클릭한 시점부터 실제 전환하기까지 걸리는 지연 시간(Delay)입니다 (($C=0$)이면 정의되지 않음).
5. $E$: 사용자가 광고를 클릭한 시점부터 학습 데이터셋을 추출한 시점까지 경과한 시간(Elapsed Time)입니다.

#### 모델 설명 
##### 기본 설정 및 가정
$$ Y = 0 \iff C = 0 \quad \text{or} \quad E < D\ $$
학습 시점에 특정 유저의 전환이 관측되지 않았다($Y=0$)는 것은 다음의 두 가지 경우 중 하나에만 해당한다는 뜻입니다.
  1. 사용자가 아예 구매할 생각이 없음 ($C=0$)
  2. 사용자가 궁극적으로 구매할 생각이 있으나($C=1$), 전환하기까지 필요한 시간이 클릭 후 현재까지 경과한 시간보다 길어서 아직 구매 버튼을 누르지 않음 ($D > E$).  

$$\Pr(C, D \mid X, E) = \Pr(C, D \mid X)$$
 사용자 특성($X$)가 주어졌을 때, '궁극적 전환 여부 ($C$)'와 '지연 시간($D$)'는 '경과 시간 ($E$)'와 서로 독립이라는 가정입니다. 즉, 클릭 후 며칠이 지났는가($E$) 는 시스템이 데이터를 추출한 시점에 의해 결정되는 수치일 뿐이며, 사용자의 본질적인 구매 성향($C$)이나 실제 구매 고민 기간($D$)에는 영향을 주지 않는다는 자명한 가정입니다.

##### 두 개의 일반화 선형 모델 (GLM)
$$ \Pr(C = 1 \mid X = x) = p(x) \quad \text{with} \quad p(x) = \frac{1}{1 + \exp(-w_c \cdot x)} $$

$$ \Pr(D = d \mid X = x, C = 1) = \lambda(x) \exp(-\lambda(x)d) \quad \text{with} \quad \lambda(x) = \exp(w_d \cdot x) \\[3pt]
= \exp(w_d \cdot x) \exp\left(-\exp(w_d \cdot x) d\right)
$$
본 논문이 제안하는 두 가지 개별 예측 모델입니다.
1. $p(x)$: Logistic Regression 모델로, 유저가 궁극적으로 전환할 확률을 예측합니다.
2. $\lambda(x)$: 전환이 발생한 경우($C=1$)의 전환시간($D$) 의 확률 분포를 나타내는 모델입니다. 전환 지연 시간 ($D$)가 양수임을 반영하기 위해 Exponential Distribution 로 지연 시간을 모델링합니다. 이때 생존 분석의 Hazard Function 역할(아직 전환하지 않은 상태에서 바로 다음 순간에 전환이 일어남) 을 하는 $\lambda(x)$ 는 양수여야 하므로 $\lambda(x) = \exp(w_d \cdot x)$ 로 매개변수화합니다. 

##### 전환이 관측된 샘플 (Positive)의 Likelihood
$$ \Pr(Y = 1, D = d_i \mid X = x_i, E = e_i) = \lambda(x_i) \exp(-\lambda(x_i)d_i) p(x_i) $$
이미 전환을 완료한 샘플($Y=1$)이 정확히 $d_i$라는 지연 시간 뒤에 전환할 확률입니다. GLM에서 구한 두 수식, "유저가 궁극적으로 전환할 확률$p(x_i)$ "과 "전환하는 유저의 지연 시간이 정확히 $d_i$ 일 확률 밀도 $\lambda(x_i)\exp(-\lambda(x_i)d_i)$"를 곱해 구합니다.  
경과 시간 $e_i$ 는 독립 가정에 의해 제거됩니다.

##### 전환이 미관측된 샘플 (Unlabeled)의 전체 확률 분해
$$ \Pr(Y = 0 \mid X = x_i, E = e_i) \\[3pt] 
= \Pr(Y = 0 \mid C = 0, X = x_i, E = e_i)\Pr(C = 0 \mid X = x_i) \\[3pt] 
+ \Pr(Y = 0 \mid C = 1, X = x_i, E = e_i)\Pr(C = 1 \mid X = x_i) $$
아직 전환이 관측되지 않은 상태($Y=0$)의 확률은 "아예 안 살 사람이 안 산 확률" 더하기 "살 사람이지만 지연 때문에 아직 안 산 확률" 로 쪼갤 수 있습니다.

###### ㄴ 살 사람이지만 지연 때문에 아직 안 샀을 확률 (Survival Function)
$$
\Pr(Y = 0 \mid C = 1, X = x_i, E = e_i) = \Pr(D > E \mid C = 1, X = x_i, E = e_i) \\[3pt] 
= \int_{e_i}^{\infty} \lambda(x) \exp(-\lambda(x)t)dt = \exp(-\lambda(x)e_i)
$$
궁극적으로 살 사람($C=1$)인데 클릭 후 지금까지 경과한 시간($e_i$) 동안 아직 안 샀을 확률을 계산합니다. 이는 실제 전환 지연 시간이 지금까지 흐른 시간보다 더 길 확률($D > E$)을 뜻합니다. $e_i$부터 무한대까지 지연 확률 밀도 함수(GLM의 두번째 식)를 적분하면 $\exp(-\lambda(x)e_i)$가 도출됩니다.

##### 전환이 미관측된 샘플 (Unlabeled)의 최종 Likelihood
$$ \Pr(Y = 0 \mid X = x_i, E = e_i) = 1 - p(x_i) + p(x_i)\exp(-\lambda(x_i)e_i) $$
위 두식에 의해 도출된 학습 데이터셋에 널려 있는 수많은 '미전환 클릭'들의 확률을 수학적으로 표현한 핵심 우도식입니다.
- $1 - p(x_i)$: 아예 전환 안 할 확률
- $p(x_i)\exp(-\lambda(x_i)e_i)$: 전환은 할 건데 고민 시간이 길어서 아직 안 보여준 확률

### Optimization
#### Expectation-Maximization
숨겨진 변수(Latent Variable)인 '사용자가 궁극적으로 전환할지 여부($C$)'를 추정하기 위해 EM (Expectation-Maximization) 알고리즘을 도입합니다.


##### 알고리즘의 배경: 잠재 변수 $C$
학습 데이터에서 사용자가 실제로 전환을 완료했다면($y_i=1$), 그 유저는 궁극적으로 전환할 사람($C_i=1$)입니다. 하지만 아직 전환하지 않았다면 ($y_i=0$), 이 유저가 아예 안 살 사람($C_i=0$)인지 아니면 살 사람인데 아직 지연 중($C_i=1, D_i > E_i$)인지 알 수 없습니다.  
따라서 진짜 정답($C$)이 숨겨져 있을 때, 이를 확률적으로 추정하면서 모델을 학습시키는 도구가 EM 알고리즘입니다.

##### E-step (Expectation, 기대 단계)

E-step의 목표는 각 샘플이 궁극적으로 전환할 사후 확률(Posterior Probability)인 $w_i$ 를 계산하는 것입니다.
$$ w_i := \Pr(C = 1 \mid X = x_i, Y = y_i, E = e_i) $$

- $y_i = 1$: 이미 전환했으므로 $w_i = 1$ 입니다.
- $y_i = 0$: 미전환 상태일 때 베이즈 정리(Bayes' Theorem)를 적용하여 아래와 같이 계산합니다.

$$ \Pr(C = 1 \mid Y = 0, X = x_i, E = e_i) = \frac{\Pr(Y = 0 \mid C = 1, X = x_i, E = e_i) \Pr(C = 1 \mid X = x_i)}{\Pr(Y = 0 \mid X = x_i, E = e_i)} $$

이 식의 분자와 분모에 Model 단계에서 구한 확률식들을 대입합니다.
- $Pr(Y = 0 \mid C = 1, X = x_i, E = e_i) = \exp(-\lambda(x_i)e_i)$  (Survival Function)
- $\Pr(C = 1 \mid X = x_i) = p(x_i)$
- $ \Pr(Y = 0 \mid X = x_i, E = e_i) = (1 - p(x_i) + p(x_i)\exp(- \lambda(x_i)e_i))$

이를 그대로 대입하여 정리하면 아래 수식이 완성됩니다. 직관적으로 보면, 클릭 후 경과 시간($e_i$)이 매우 짧다면 분자의 지수 파트가 1에 가까워져 $w_i$ 가 커집니다. 즉, "아직 클릭한 지 얼마 안 되었으니 살 사람인데 지연 중일 확률($w_i$)이 높다"고 판단합니다.  
반대로 클릭 후 한 달이 지나도록($e_i \to \infty$) 소식이 없다면 분자가 0에 수렴하여 $w_i \approx 0$ 이 되고, 모델은 이 샘플을 확실한 부정 샘플(안 살 사람)로 취급하게 됩니다.

$$ w_i = \frac{p(x_i)\exp(-\lambda(x_i)e_i)}{1 - p(x_i) + p(x_i)\exp(-\lambda(x_i)e_i)} $$
 
#####  M-step (Maximization, 최대화 단계)

M-step의 목표는 E-step에서 구한 사후 확률(가중치) $w_i$를 고정한 상태에서, 전체 데이터의 Expected Log-Likelihood 를 가장 크게 만드는 파라미터 $p(x)$의 가중치 $w_c$와 $\lambda(x)$ 함수의 가중치 $w_d$를 찾는 것입니다.

###### 기대 로그 우도의 정의
$$
\sum_{i, y_i=1} \log \Pr(Y = 1, D = d_i \mid X = x_i, E = e_i) \\[3pt] 
+ \sum_{i, y_i=0} \left[ (1 - w_i) \log \Pr(Y = 0, C = 0 \mid X = x_i, E = e_i) \\[3pt]
+ w_i \log \Pr(Y = 0, C = 1 \mid X = x_i, E = e_i) \right]
$$
- 전환된 샘플($y_i=1$)은 잠재 변수가 $C=1$ 로 확실하므로 일반 로그 우도를 더합니다.
- 전환되지 않은 샘플($y_i=0$)은 실제 상태가 $C=0$일 확률 $1-w_i$ 과 $C=1$ 일 확률 $w_i$ 로 가중평균된 로그 우도를 반영합니다.

<b>  ㄴ 미전환 샘플의 기대 로그 우도 전개 </b>  
위 의 오른쪽 항(미전환 샘플 부분)의 결합 확률을 수학적으로 풀어서 전개한 식입니다.
1. $\Pr(Y = 0, C = 0 \mid X, E) = \Pr(Y = 0 \mid C = 0, X, E)\Pr(C = 0 \mid X) = 1 \cdot (1 - p(x_i)) = 1 - p(x_i)$
2. $\Pr(Y = 0, C = 1 \mid X, E) = \Pr(Y = 0 \mid C = 1, X, E)\Pr(C = 1 \mid X) = \exp(-\lambda(x_i)e_i) \cdot p(x_i)$

위 두식을 대입해 기대 로그 likelihood를 정리하면 아래와 같습니다.
$$
(1 - w_i) \log(1 - p(x_i)) + w_i \left[ \log(p(x_i)) - \lambda(x_i)e_i \right]
$$

###### 최종 M-step 목적 함수
Model에서 구한 전환 관측 likelihood 와 위에서 구한 미전환 기대 로그 likelihood를 기대 로그 likelihood 수식에 모두 대입해서 정리하면 아래와 같습니다.
$$
\sum_i \left[ w_i \log p(x_i) + (1 - w_i) \log(1 - p(x_i)) \right] + \sum_i \left[ \log(\lambda(x_i))y_i - \lambda(x_i)t_i w_i \right]
$$
단, $t_i$ 는 $y_i = 1$ 이면 지연 시간 $d_i$ 가 되고, $y_i = 0$ 이면 경과 시간 $e_i$ 가 됩니다.

앞서 다룬 EM 알고리즘은 수학적으로 아름답게 분리되지만, 하나의 M-step을 밟을 때마다 내부에서 로지스틱 회귀와 지수 회귀를 완전히 수렴할 때까지 매번 새로 풀어야 하는 '중첩된 최적화(Nested Optimization)' 구조 때문에 대규모 데이터 학습 시 속도가 매우 느리다는 치명적인 단점이 있습니다. 

이를 해결하기 위해 논문의 저자들은 잠oint optimization방식을 제안합니다.

#### Joint optimization
##### 목적 함수 (Regularized Negative Log-Likelihood)
EM 알고리즘은 M-step을 밟을 때마다 내부적으로 로지스틱 회귀와 지수 회귀 최적화를 수렴할 때까지 매번 새로 돌려야 하므로 학습 속도가 매우 느리다는 단점이 있습니다. 이 때문에 논문의 저자들은 실제 대규모 실험에서 $p(x)$의 가중치 $w_c$와 $\lambda(x)$ 함수의 가중치 $w_d$를 에 대해 동시에 최적화할 수 있는 하나의 단일 목적 함수를 정의합니다.

과적합(Overfitting)을 방지하기 위해 L2 Regularization를 포함한 Regularized Negative Log-Likelihood 를 최소화하는 문제로 정의합니다.

$$ \arg \min_{w_c, w_d} \mathcal{L}(w_c, w_d) + \frac{\mu}{2} \left( \|w_c\|_2^2 + \|wd\|_2^2 \right) $$

- $\mathcal{L}(w_c, w_d)$: 전체 데이터에 대한 Negative Log-Likelihood 입니다.
- $\frac{\mu}{2} \left( \|w_c\|_2^2 + \|wd\|_2^2 \right)$: 모델이 가질 수 있는 가중치들의 크기를 제한하는 L2 Regularization 입니다. 이 항을 최소화함으로써 특정 피처에 가중치가 과도하게 몰려 오버핏되는 현상을 막아줍니다.

###### ㄴ Negative Log-Likelihood 수식
$$
\mathcal{L}(w_c, w_d) = - \sum_{i, y_i=1} \left[ \log p(x_i) + \log \lambda(x_i) - \lambda(x_i)d_i \right] - \sum_{i, y_i=0} \log \left[ 1 - p(x_i) + p(x_i) \exp(-\lambda(x_i)e_i) \right] \\[3pt]
\text{where} \quad p(x) = \frac{1}{1 + \exp(-w_c \cdot x)}, \quad \lambda(x) = \exp(w_d \cdot x)
$$

이전에 구한 전환 관측 우도와 미전환 관측 우도에 각각 로그를 취하고 음수 부호를 붙인 뒤, 전체 데이터에 대해 단순 합산한 형태입니다. EM 알고리즘처럼 가중치 $w_i$를 계산하여 대입하는 단계를 거치지 않고, $d_i, e_i, y_i$ 만으로 구성된 단일 로그 우도 함수를 직접 도출한 것입니다.

이 목적 함수는 비록 비볼록 함수이지만, 제약 조건이 없고(Unconstrained) 모든 영역에서 미분 가능하며, 심지어 2차 미분까지 가능(Twice Differentiable)한 아주 유순한 수학적 성질을 지니고 있습니다. 따라서 저자들은 경사하강법 계열 중 대규모 최적화 문제에서 가장 빠르고 효율적이라고 알려진 Quasi-Newton 기반의 L-BFGS 알고리즘 을 사용하여 두 파라미터 $w_c$와 $w_d$를 동시에 최적화하는 데 성공하였습니다.

#### 실제 광고 서빙 (eCPM 계산)
실시간 경매(RTB) 시스템이 입찰 가치(eCPM)를 평가할 때는 $\Pr(\text{conversion} \mid \text{click})$자리에 우리가 학습시킨 전환 예측 모델 $p(x)$가 그대로 적용됩니다.
$$
\text{eCPM} = \text{CPA} \times \Pr(\text{click}) \times p(x)
$$

반면 지연 예측 모델 $\lambda(x)$는 사용자가 클릭 후 실제 전환에 이르기까지 걸리는 시간적 분포를 나타냅니다. 그렇기 때문에 $\lambda(x)$ 는 오직 학습 단계에서 아직 전환이 관측되지 않은 데이터($Y=0$)를 만났을 때, "이 유저는 아예 안 살 유저인가, 아니면 살 유저인데 단지 아직 안 산 것뿐인가?"를 판단하는 도구 로만 기여하고, 학습이 완료되면 서빙 환경에는 탑재되지 않고 버려집니다.

>Once these two models are trained, the former [$p(x)$] is used to predict the probabilities of conversion while the latter [$\lambda(x)$] is discarded."

(두 모델이 학습 완료되면, 전자는 전환 확률을 예측하는 데 사용되고 후자는 버려진다.)


https://dl.acm.org/doi/10.1145/2623330.2623634

## A Nonparametric Delayed Feedback Model
https://arxiv.org/abs/1802.00255

## Addressing Delayed Feedback for Continuous Training
https://arxiv.org/abs/1907.06558


## References
### Tier 0 — 배경

| 인용 | 논문 | 왜 중요한가 |
|---:|---|---|
| **556** | [ESMM: Entire Space Multi-Task Model](https://arxiv.org/abs/1804.07931) (SIGIR'18) | 지연 문제는 아니지만 **CVR 예측의 sample selection bias / data sparsity**를 정의한 논문. 이 분야 모든 논문이 베이스라인으로 씁니다. 후에 ESDF'23이 이걸 지연 문제와 합칩니다 |
| **209** | [Modeling Delayed Feedback in Display Advertising](https://dl.acm.org/doi/10.1145/2623330.2623634) (Chapelle, KDD'14) | **원조**. CVR × 지수분포 지연을 EM으로 동시 추정. 이후 전부 이 논문의 "파라메트릭 지연 가정"과 "배치 학습 전제"를 깨는 과정 |
| **110** | [Stochastic Bandit Models for Delayed Conversions](https://arxiv.org/abs/1706.09186) (UAI'17) | 같은 문제의 **밴딧/온라인 학습 버전**. 예측이 아니라 의사결정으로 갈 때 참조점 |
| **57** | [A Nonparametric Delayed Feedback Model](https://arxiv.org/abs/1802.00255) (2018) | Chapelle의 지수분포 가정을 **비모수**로 대체. "지연 분포를 직접 모델링" 계열의 마지막 정점 — 이후 계보가 importance weighting으로 갈라짐 |

### Tier 1 — 계보의 척추. 필수 5편

| 인용 | 논문 | 움직인 축 |
|---:|---|---|
| **68** | [Ktena et al., Addressing Delayed Feedback for Continuous Training](https://arxiv.org/abs/1907.06558) (**RecSys'19**) | 지연 분포 모델링 → **분포 편향 보정**. FNC(출력) / FNW(손실) 두 갈래 제시 |
| **45** | [ES-DFM: Elapsed-Time Sampling](https://arxiv.org/abs/2012.03245) (AAAI'21) | 대기 창 `w`를 **설계 변수**로 승격. 계보에서 가장 자주 재현·비교되는 기준선 |
| **43** | [DEFER: Real Negatives Matter](https://arxiv.org/abs/2104.14121) (KDD'21) | 가중치 설계 → **proposal 분포 설계**. 전부 duplicate해 샘플 공간 대칭화 |
| **40** | [FSIW: A Feedback Shift Correction](https://arxiv.org/abs/2002.02068) (WWW'20) | 문제를 **counterfactual 분포 시프트**로 정식화, 불편성 증명 |
| **34** | [DEFUSE: Asymptotically Unbiased Estimation via Label Correction](https://arxiv.org/abs/2202.06472) (WWW'22) | negative 한 덩어리 → **4종 분해**(IP/FN/RN/DP). ES-DFM·DEFER를 하나의 틀로 통합·개선한 편. **파일에 빠진 것 중 1순위** |

**5편만 읽는다면 이 표 그대로**, 순서는 Ktena → FSIW → ES-DFM → DEFER → DEFUSE.


### Tier 2 — 중요 확장

| 인용 | 논문 | 특징 | 저자·계열 연속성 |
|---:|---|---|---|
| **17** | [ULC](https://arxiv.org/abs/2307.12756) (KDD'23) | 보조 모델로 음성의 reversal 추정 → **counterfactual labeling + 비편향 corrected loss** | FSIW·nnDF·DEFUSE의 라벨 보정 갈래 |
| **16** | [ESDF](https://arxiv.org/abs/2308.04768) (CIKM'23) | ESMM × 지연 보정 (entire space + cascade) | — |
| **15** | [GDFM](https://arxiv.org/abs/2206.00407) (NeurIPS'22) | 전환/경과시간만 보지 않고 **post-click 행동을 확률적 피드백으로**. temporal gap + sampling gap 동시 보정 | **ES-DFM 저자(Yang·Zhan)의 직접 확장** ✓ |
| **15** | [Follow the Prophet](https://arxiv.org/abs/2108.06167) (SIGIR'21) | 지연 분포를 시계열 예측 | 다중창/앙상블 갈래의 시작 |
| **14** | [DDFM](https://doi.org/10.1145/3583780.3614856) (CIKM'23) | 최신 스트리밍 샘플과 라벨 확정 샘플을 **별도 분포로 취급, 두 비편향 추정기 결합** | DEFER가 남긴 신선도↔정확도 충돌을 정면 해결 |
| **10** | [Kato & Yasui, Time Window Assumption](https://arxiv.org/abs/2009.13092) (KDD'22) — `convDF`/`nnDF` | 시간창 가정 아래 **전 샘플을 쓰는 볼록·비편향 경험위험** + 음수 위험 방지 **non-negative 보정(`nnDF`)** | **FSIW 저자 Yasui 참여 — 가장 명확한 저자 연속성** |
| **8** | [Neural Satellite Networks](https://doi.org/10.1145/3539618.3591747) (SIGIR'23) | main + satellite 앙상블 | 앙상블 갈래 |
| **6** | **[MISS](https://ojs.aaai.org/index.php/AAAI/article/view/28726) (AAAI'24)** ⭐ | 짧은 창~긴 창 **여러 구간별 예측 head + 경량 synthesizer로 결합**. 고정 관측창 선택을 아예 회피 | **다중 관측창 계열의 핵심**. 저자 Xiang Ao → SIGIR'26 TRACE로 이어짐 |
| — | [Dual Learning Algorithm](https://arxiv.org/pdf/1910.01847) (SIGIR'20) | 지연 분포 × CVR dual 학습 | FSIW와 같은 팀 계열 |

### Tier 3 — 최신

| 인용 | 논문 | 왜 |
|---:|---|---|
| — | [CVR 예측 서베이](https://arxiv.org/abs/2512.01171) (2025-12) | 6개 카테고리 분류 + debiased CVR 방향 제시 |
| **2** | [IF-DFM](https://arxiv.org/abs/2502.01669) (AAAI'26) | 샘플 재생(DEFER 계열)을 넘어 **새 피드백의 파라미터 영향 자체를 influence function으로 계산**. inverse HVP를 최적화 문제로 재정식화 |
| **1** | [Personalized Interpolation](https://arxiv.org/abs/2501.14103) (CIKM'25, Meta) | 짧은 창/긴 창 예측을 광고주·트래픽 특성에 맞게 보간 → **개인화된 유효 관측창**. FTP·MISS를 프로덕션 지연 패턴에 맞춘 확장 |
| — | [Follow the TRACE](https://arxiv.org/abs/2604.23197) (SIGIR'26) | 누적 post-click 행동을 **feedback trajectory**로. 미확정 샘플에 동적 posterior + 신뢰도 기반 보완 |
| — | [TWICE](https://arxiv.org/abs/2607.25404) (2026-07) | **click clock / conversion clock 분리**. 짧은 창으로 학습해 긴 목표창 CVR + 지연 CDF 동시 추정, 하나의 CDF에서 다중 horizon 단조 예측 |
| — | [TESLA / CASCADE](https://arxiv.org/abs/2601.19965) (WWW'26) | 다단계 지연(전환→환불) + 공개 데이터셋 |
| — | [READER / TRACE](https://arxiv.org/abs/2601.20307) (WWW'26) | 연속값 GMV로 대상 확장 |
| — | [MAC / PyMAL](https://arxiv.org/abs/2603.02184) (KDD'26) | 다중 어트리비션 라벨 동시 학습 |

### 추가된 논문들 때문에 드러난 구조: 계보가 5개 지류로 갈라집니다

Tier로만 보면 안 보이는데, MISS와 TWICE가 들어오니 **다중 관측창 계열**이 독립 지류로 확실해집니다.

```
Chapelle'14 / Nonparametric'18   (지연 분포 직접 모델링)
        │
        ▼
   Ktena'19 (FNW/FNC)  ─── 문제를 "분포 편향 보정"으로 재정의
        │
        ├─① 가중치·추정량 보정
        │    FSIW'20 → ES-DFM'21 → DEFER'21 → DEFUSE'22 → nnDF'22
        │
        ├─② 라벨/파라미터 직접 보정
        │    (DEFUSE에서 분기) → ULC'23 → IF-DFM'25
        │
        ├─③ 다중 관측창·앙상블          ★ MISS 추가로 명확해진 지류
        │    FTP'21 → NSN'23 / DDFM'23 → MISS'24 → PI'25 → TWICE'26
        │
        ├─④ post-click 행동·궤적
        │    GDFM'22 (ES-DFM 저자) → TRACE'26 (MISS 저자)
        │
        └─⑤ 문제 정의 확장
             ESMM'18 × ESDF'23 → TESLA'26(다단계) / READER'26(연속값) / MAC'26(다중 어트리비션)
```

지류별 한 줄 요약:

- **①** 창은 하나로 고정, **추정량을 정교하게** → 이론이 가장 단단함 (`nnDF`가 종착점)
- **②** 가중치 대신 **라벨/파라미터를 건드림** → 보조 모델 의존을 줄이는 방향
- **③** 창을 하나로 고르는 걸 **포기**하고 여러 개를 동시에 → 산업 배포에 가장 강함 (TWICE의 A/B +2.5%)
- **④** 라벨이 안 왔어도 **중간 행동이 정보를 준다**
- **⑤** 애초에 예측 대상 자체를 바꿈

### 읽는 순서 제안 (지류별)

**이론 중심**이면: Tier 1 5편 → `nnDF` → ULC → IF-DFM (①②)
**프로덕션 중심**이면: Ktena → ES-DFM → DEFER → **MISS** → Personalized Interpolation → TWICE (③)
**최신 방향 탐색**이면: 서베이(2512.01171) → GDFM → TRACE → TESLA/READER (④⑤)

### 데이터셋 / 재현 기준

논문 비교할 때 필요한 실무 정보입니다.

- **Criteo Sponsored Search Conversion Logs** — 지연 시간이 붙은 사실상의 표준 벤치마크. Tier 1~2 전부 여기서 비교합니다
- **Taobao / Alibaba 계열** — ES-DFM, DEFER 등 산업 논문의 프로덕션 데이터
- 신규 공개: [CASCADE](https://arxiv.org/abs/2601.19965)(NetCVR), [alimama-tech/OnlineGMV](https://github.com/alimama-tech/OnlineGMV)(GMV), [alimama-tech/PyMAL](https://arxiv.org/abs/2603.02184)(multi-attribution)


## 계보 그룹 요약
| 그룹 | 소속 | 편수 | 논문 | 담당 지류 | 핵심 인물 |
|---|---|---:|---|---|---|
| **A** | Alibaba / Alimama | **7** | ESMM'18, DEFER'21, DEFUSE'22, ESDF'23, TESLA·READER·MAC'26 | ①⑤ | **Xiang-Rong Sheng (5편)**, Jian Xu, Bo Zheng, Han Zhu |
| **B** | 중국과학원 ICT | **4** | FTP'21, NSN'23, MISS'24, TRACE'26 | ③④ | **Xiang Ao (4편)**, Qing He |
| **C** | 난징대 LAMDA | **2** | ES-DFM'21, GDFM'22 | ①→④ | Jia-Qi Yang, De-chuan Zhan |
| **D** | CyberAgent | **3** | Dual Learning'20, FSIW'20, nnDF'22 | ① (이론) | **Shota Yasui (3편)** |
| 단발 | Criteo, 치바공대, **Twitter**, RUC, Tsinghua+Huawei, **Meta**, USTC, **Kuaishou** | 8 | — | ②③ 위주 | — |