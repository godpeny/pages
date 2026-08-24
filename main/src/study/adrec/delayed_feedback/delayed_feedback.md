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
### 문제점
기존에는 Delayed Feedback Model간 지연 분포가 지수 분포(Exponential Distribution)를 따른다고 가정하지만 현실에서 지연 시간이 특정 지수 분포를 따른다는 보장은 없으며, 데이터의 특성에 따라 최적의 지연 시간 분포 형태가 다릅니다.

### 제안: 비모수적 지연 피드백 모델 (NoDeF)
분포 가정이 없는 모델 설계: 본 논문에서는 지수 분포나 와이블(Weibull) 분포 같은 특정한 모수적 분포(Parametric Distribution)를 사전에 가정하지 않고 시간 지연을 표현하는 모델을 제시합니다.고정된 수학적 분포를 억지로 끼워 맞추는 대신, 노출된 광고의 콘텐츠와 사용자의 다양한 특성(Feature)에 대응하여 시간 지연 분포를 학습합니다.

### Model
NoDeF는 크게 두 가지 확률 모델의 결합으로 구성됩니다.
1. 시간 지연 모델 (Time Delay Model): 광고 클릭 후 구매 전환까지 걸리는 시간 지연을 모델링.
2. 전환 모델 (Conversion Model): 새로운 사용자에게 광고가 노출되었을 때 궁극적으로 전환할지 여부를 예측하는 이진 분류기.

#### 변수 정의
- 확률 밀도 함수 (PDF) $f(t)$: 특정 시점 $t$에 정확히 이벤트(구매 전환)가 발생할 순간적인 확률 밀도를 뜻합니다.
- 누적 분포 함수 (CDF) $F(t)$: 처음(시점 0)부터 특정 시점 $t$까지의 누적 기간 동안 이벤트가 발생할 확률의 합입니다. $F(t) = \int_{0}^{t} f(x)dx $
- 생존 함수 (Survival Function) $s(t)$: 시점 $t$가 될 때까지 이벤트가 발생하지 않고 '생존'(미전환 상태로 유지)해 있을 확률을 뜻합니다. 전체 확률 1에서 이미 전환이 완료된 누적 확률을 빼서 구합니다.  $s(t) = 1 - F(t) $
- 위험 함수 (Hazard Function) $h(t)$: 시점 $t$ 직전까지는 아직 이벤트가 발생하지 않은 상태에서, 정확히 그 시점 $t$에 도달하는 순간 이벤트(구매 완료)가 터질 비율을 뜻합니다. 특정 시점의 확률 밀도 함수 $f(t)$를 그 시점까지 아직 이벤트가 안 일어나고 버텼을 생존 확률 $s(t)$로 나누어 정의합니다. $h(t) = \frac{f(t)}{s(t)} $
  - 위험 함수 $h(t)$만 정의하면, 적분을 통해 생존 함수 $s(t)$를 유도할 수 있습니다.  
  $s(t) = \exp \left( - \int_{0}^{t} h(x)dx \right)$

### 비모수적 위험 함수의 정의 
#### 비모수적 위험 함수 (Hazard Function)
$$
h(d_i; x_i, V) = \sum_{l=1}^L \alpha_l(x_i; V) k(t_l, d_i)
$$
$i$ 번째 샘플(유저 및 광고 피처 $x_i$)의 시간 지연 $d_i$ 시점에서의 순간적인 전환율을 정의합니다. 시간축 위에 일정한 간격으로 배치된 $L$ 개의 의사 포인트(Pseudo-points) $t_l$ 을 설정합니다. 그리고 각 의사 포인트에서의 지연 가중치(강도)를 나타내는 강도 함수($\alpha_l(x_i; V)$)와 전환 시점과의 시간적 유사도를 측정하는 커널 함수 ($k(t_l, d_i)$)를 곱해 모두 더한(가중합) 형태입니다.  
(커널 밀도 추정(KDE)의 아이디어를 차용하되, 모든 관측 데이터를 계산하는 대신 고정된 \\(L\\)개의 점만 계산하므로 연산이 매우 빠릅니다.)

<b> 가우시안 커널 (Gaussian Kernel) </b>  
$$ k(t_l, \tau) = \exp \left( -\frac{(t_l - \tau)^2}{2h^2} \right) $$
* **설명**: 대역폭(Bandwidth) $h > 0$ 를 가진 가우시안 커널 함수입니다. 의사 포인트($t_l$) 과 실제 지연 시간 ($\tau$ 가 시간상으로 가까울수록 큰 값을 가집니다.

<b> $[0, a]$ 구간에서의 커널 적분값 (Analytical Integration) </b>   
$$ 
\int_{0}^{a} k(t_l, \tau)d\tau = -h \frac{\sqrt{\pi}}{2} \left[ \text{erf} \left( \frac{t_l - a}{\sqrt{2}h} \right) - \text{erf} \left( \frac{t_l}{\sqrt{2}h} \right) \right] 
$$
오차 함수(Error Function, $\text{erf}$)를 사용하여 가우시안 커널을 시간 $[0, a]$까지 적분한 값입니다. 복잡한 수치적 근사 없이 수학적으로 정확하고 빠르게 적분값을 얻을 수 있습니다.

<b> $[a, \infty)$ 구간에서의 커널 적분값 </b>  
$$
\int_{a}^{\infty} k(t_l, \tau)d\tau = h \frac{\sqrt{\pi}}{2} \left[ 1 + \text{erf} \left( \frac{t_l - a}{\sqrt{2}h} \right) \right]
$$
마찬가지로 가우시안 커널을 특정 시점 $[a, \infty)$ 적분한 값입니다.

<b> 강도 함수 (Intensity Function) </b>  
$$ \alpha_l(x_i; V) = \left( 1 + \exp \left( -V_l^T x_i \right) \right)^{-1} $$

가상의 의사 포인트 $t_l$ 에 부여될 가중치(강도)를 계산하는 함수입니다. 유저 및 광고의 입력 특징 벡터 $x_i$ 와 매개변수 행렬 $V$의 $l$ 번째 행 벡터 $V_l$ 의 결합을 시그모이드(Sigmoid) 함수에 통과시켜 구합니다. 시그모이드를 사용하므로 $\alpha_l$ 의 값은 항상 0에서 1 사이가 되며, 이는 위험 함수 $h(t)$가 물리적으로 늘 양수를 유지할 수 있도록 합니다.

### 생존 함수와 전환 사건 확률 
#### 생존 함수 (Survival Function)
$$
s(d_i; x_i, V) = \exp \left( -\int_{0}^{d_i} h(\tau; x_i, V) d\tau \right) = \exp \left( -\sum_{l=1}^{L} \alpha_l(x_i; V) \int_{0}^{d_i} k(t_l, \tau) d\tau \right)
$$
$d_i$ 시점까지 사용자가 아직 구매(전환)하지 않고 미전환 상태로 버티고 있을 확률입니다. 앞서 구한 비모수적 위험 함수 $h(t)$ 를 생존 분석의 변환 공식,$s(t) = \exp \left( - \int_{0}^{t} h(x)dx \right)$ 에 대입하여 도출했습니다.

#### **NoDeF의 시간 지연 모델**
$$
p(d_i \mid x_i, c_i = 1) = s(d_i; x_i, V) h(d_i; x_i, V)
$$
궁극적으로 구매할 유저($c_i = 1$)가 정확히 클릭 후 $d_i$ 시점에 구매를 완료할 확률 밀도입니다.  
관계식 $f(t) = s(t)h(t)$ 를 따른 것으로, $d_i$ 시점까지 구매하지 않고 버텼을 확률($s$)과 바로 그 시점 $d_i$에 구매를 터뜨릴 확률($h$) 의 곱으로 정의됩니다.

#### **NoDeF의  전환 모델**
궁극적으로 전환이 일어날 사건을 나타내는 잠재 변수(Hidden Variable)를 $c_i \in \{0, 1\}$ 라고 할 때, 본 논문에서는 로지스틱 회귀를 사용하여 예측합니다.
$$
p(c_i = 1 \mid x_i; w) = \left( 1 + \exp(-w^T x_i) \right)^{-1} \\[3pt]
p(c_i = 0 \mid x_i; w) = 1 - p(c_i = 1 \mid x_i)
$$
유저 및 광고 피처 $x_i$ 와 가중치 벡터 $w$를 결합하여, 해당 유저가 시간과 상관없이 언젠가는 최종적으로 전환할 확률을 예측합니다.

#### **Joint Model**
최종적으로 NoDeF는 이 두 모델을 결합하여, "어떤 유저가 궁극적으로 구매할 확률"과 "구매한다면 $d_i$라는 시간 지연 뒤에 구매할 확률"을 곱한 하나의 결합 확률(Joint Probability) 형태로 데이터를 해석합니다. 즉, 궁극적으로 살 유저($c_i = 1$)가 클릭 후 정확히 $d_i$ 시점에 전환을 완료할 결합 확률은 다음과 같이 두 모델식의 곱셈으로 표현됩니다.
$$ p(c_i = 1, d_i \mid x_i; \Theta) = \underbrace{p(c_i = 1 \mid x_i; w)}_{\text{전환 모델}} \times \underbrace{p(d_i \mid x_i, c_i = 1; V)}_{\text{시간 지연 모델}} \\[3pt]
= p(c_i = 1 \mid x_i; w) \times \Big[ s(d_i; x_i, V) h(d_i; x_i, V) \Big] $$


### 목적함수
$w$와 $V$를 역산해 내기 위한 목적 함수를 구합니다.
#### 관측 데이터의 우도 함수 정의  (Likelihood of Observation)
$$
I_1 = \{i \mid y_i = 1, i = 1, 2, \dots, n\} \\[3pt]
I_0 = \{i \mid y_i = 0, i = 1, 2, \dots, n\} 
$$
전체 데이터를 관측 기간 내에 실제 전환이 완료된 데이터($I_1$)와 전환이 관측되지 않은 데이터($I_0$)로 나눕니다.

#### 전체 데이터 관측 우도
$$
p(D; \Theta) = \prod_{i=1}^n \sum_{c_i \in \{0, 1\}} p(y_i \mid x_i, c_i, e_i) p(c_i \mid x_i) p(d_i \mid x_i, c_i = 1)
$$
1. $p(c_i \mid x_i)$ : 사용자의 특징 특징 벡터($x_i$)가 주어졌을 때, 이 사용자가 시간과 관계없이*궁극적으로 전환을 일으킬 확률입니다 (로지스틱 회귀로 예측).
2. $p(d_i \mid x_i, c_i = 1)$ : 만약 사용자가 궁극적으로 전환할 사람($c_i = 1$)이라면, 클릭 후 정확히 $d_i$ 만큼의 시간 지연 후에 전환을 완료할 확률 밀도입니다.
3. $p(y_i \mid x_i, c_i, e_i)$: 궁극적 전환 여부($c_i$)와 경과 시간($e_i$)이 주어졌을 때, 데이터 마감 시점에 우리 시스템에 실제 관측 결과($y_i \in \{0, 1\}$)로 기록될 조건부 확률입니다.
4. $D = \{(x_i, y_i, d_i, e_i)\}_{i=1}^n$: 광고 로그 시스템에서 수집된 전체 관측 데이터셋(Observation Set)
5. $\Theta = \{V, w\}, V \in \mathbb{R}^{L \times M}, w \in \mathbb{R}^M $: 학습을 통해 찾아내야 하는 모델의 전체 파라미터 집합입니다. 

<b> 일관성 보증 조건 (Consistency Conditions) </b>  
궁극적으로 전환하지 않을 사람($c_i = 0$)의 상태적 특성을 정의합니다.
$$
p(y_i = 0 \mid x_i, c_i = 0, e_i) = 1 \\[3pt]
p(y_i = 1 \mid x_i, c_i = 0, e_i) = 0
$$
최종 전환 의사가 없는 유저($c_i = 0$)라면 경과 시간($e_i$)에 상관없이 무조건 관측 라벨은 미전환($y_i = 0$)이어야 하며, 실제 전환 완료($y_i = 1$)가 관측될 확률은 절대 있을 수 없습니다.

#### 조건 분해를 통한 우도 함수의 변형
$$
p(D; \Theta) = \left[ \prod_{i \in I_1} \sum_{c_i \in \{0, 1\}} p(y_i \mid x_i, c_i, e_i) p(c_i \mid x_i) p(d_i \mid x_i, c_i = 1) \right] \\[3pt]\times \left[ \prod_{i \in I_0} \sum_{c_i \in \{0, 1\}} p(y_i \mid x_i, c_i, e_i) p(c_i \mid x_i) p(d_i \mid x_i, c_i = 1) \right]
$$
전체 데이터셋을 실제로 전환이 완료된 긍정 샘플 그룹($I_1$)과 아직 미전환 상태인 부정 샘플 그룹($I_0$)으로 나눕니다. 

<b> 긍정 샘플 그룹 ($i \in I_1$), 즉 실제 전환 관측값 ($y_i = 1$)인 경우 </b>  
이 그룹은 이미 실제로 구매를 하여 $y_i = 1$ 로 관측된 상태입니다. 시그마 식의 $y_i$ 자리에 1을 대입하고, $c_i = 0$ 과 $c_i = 1$ 일 때의 합으로 풉니다.

$$
\sum_{c_i \in \{0, 1\}} p(y_i = 1 \mid x_i, c_i, e_i) p(c_i \mid x_i) p(d_i \mid x_i, c_i = 1) \\[3pt]
= \underbrace{p(y_i = 1 \mid x_i, c_i = 0, e_i) p(c_i = 0 \mid x_i) p(d_i \mid x_i, c_i = 1)}_{c_i = 0 \text{ 인 경우}} \\[3pt]
+ \underbrace{p(y_i = 1 \mid x_i, c_i = 1, e_i) p(c_i = 1 \mid x_i) p(d_i \mid x_i, c_i = 1)}_{c_i = 1 \text{ 인 경우}}
$$

- 첫 번째 항 ($c_i = 0$): 일관성 조건 식 $p(y_i = 1 \mid x_i, c_i = 0, e_i) = 0$ 이므로, 항 전체가 0이 되어 증발합니다.
- 두 번째 항 ($c_i = 1$): 실제 구매 완료 상태가 관측되었으므로 이 조건부 관측 확률 $p(y_i = 1 \mid x_i, c_i = 1, e_i)$ 은 1이 됩니다. 또한, 실제 수집된 지연 시간이므로 지연 확률 밀도를 $p(d_i \mid y_i = 1, x_i)$로 표기할 수 있습니다.

따라서, $I_1$ 그룹을 정리하면 아래와 같습니다.
$$ \prod_{i \in I_1} p(c_i = 1 \mid x_i) p(d_i \mid y_i = 1, x_i) $$


<b> 부정/미전환 샘플 그룹 ($i \in I_0$), 즉 실제 전환 관측값 ($y_i = 0$) 인 경우 </b>  
이 그룹은 관측 종료 시점까지 구매하지 않은 상태($y_i = 0$)입니다. 이들은 진짜 안 살 사람($c_i = 0$)일 수도 있고, 살 예정인데 시간만 지연되고 있는 사람($c_i = 1$)일 수도 있습니다. 다만, 이 유저들은 아직 전환하지 않았기 때문에 실제 지연 시간 데이터인 $d_i$ 를 현실적으로 관측할 수 없습니다 ($d_i = \infty$). 데이터상에 관측되지 않은 임의의 $d_i$ 에 대해 확률 주변화(Marginalization)를 거치게 되므로, 특정 시점의 단일 확률 밀도 함수인 $p(d_i \mid x_i, c_i = 1)$ 항은 계산에서 제외됩니다.   
($p(y_i = 0 \mid x_i, c_i=1, e_i)= \int_{e_i}^{\infty} \underline{p(d_i = \tau \mid x_i, c_i=1)} d\tau$)

그 결과, 부정 샘플 그룹 $I_0$에 대한 우도는 다음과 같이 단순화됩니다.
$$
\prod_{i \in I_0} \sum_{c_i \in \{0, 1\}} p(y_i = 0 \mid x_i, c_i, e_i) p(c_i \mid x_i)
$$

#### 목적함수 식의 완성 
$$
p(D; \Theta) = \prod_{i \in I_1} p(c_i = 1 \mid x_i) p(d_i \mid y_i = 1, x_i) \times \prod_{i \in I_0} \sum_{c_i \in \{0, 1\}} p(y_i = 0 \mid x_i, c_i, e_i) p(c_i \mid x_i)
$$

<b> 미전환 상태의 확률 분해 </b>  
미전환 그룹 중 '살 예정인데 지연되는 사람($c_i = 1$)'이 관측 마감 시점 $e_i$ 까지 여전히 미전환($y_i = 0$) 상태로 남아있을 확률을 수학적으로 전개하는 과정입니다.

<b> 지연 시간이 경과 시간보다 클 확률 </b>  
$$
p(y_i = 0 \mid x_i, c_i = 1, e_i) = p(d_i > e_i \mid x_i, c_i = 1, e_i) \\[3pt]
= 1 - \int_{0}^{e_i} p(d_i = \tau \mid c_i = 1, x_i) d\tau
$$
실제 구매를 행하기까지 걸릴 지연 시간($d_i$)가, 광고 클릭 후 현재까지 흐른 관찰 시간($e_i$) 보다 더 길기 때문에 아직 미전환 상태로 관측되는 것임을 수학적으로 명시한 것입니다.  
$ d_i > e_i $ 가 발생할 확률은, 전체 확률 1에서 '처음부터 경과 시점 $e_i$ 까지의 사이에 이미 구매가 완료되었을 누적 확률'을 차감한 여사건의 확률과 같습니다.

<b> 최종 생존 함수로의 수렴 </b>  
$$ p(y_i = 0 \mid x_i, c_i = 1, e_i) = s(e_i; x_i, V) $$
$s(t) = 1 - F(t)$의 정의에 따라, 위 식의 구조는 정확히 경과 시간 $e_i$ 시점에서의 생존 함수 $s(e_i; x_i, V)$ 와 일치하게 됩니다. 결과적으로 미관측 유저의 우도는 복잡한 적분 연산 없이, 우리가 정의한 생존 확률 식 $s(e_i; x_i, V)$ 를 그대로 대입하여 구동할 수 있습니다.

### Learning Algorithm
궁극적으로 최적화하려는 대상은 관측된 데이터의 로그 우도인 "목적함수 식의 완성"의 수식의 로그 수식입니다. 하지만 로그 안에 덧셈($\log(A + B$))이 들어가 있으면, 이를 가중치 벡터 $w$ 나 $V$ 로 편미분할 때 분수 형태의 매우 복잡한 연쇄 법칙(Chain Rule)이 생기며 결합 연산이 얽히게 됩니다. 컴퓨터가 경사하강법으로 기울기(Gradient)를 구하는 것이 사실상 불가능해집니다.
직접 계산하기 불가능한 실제 로그 우도(Log-likelihood) 함수를 대신하여 컴퓨터가 극대화할 수 있는 '수학적 하한선(Lower Bound)'을 젠센의 부등식을 이용하여 정의합니다. 

$$ \log p(D; \Theta) = \sum_{i \in I_1} \log \left[ p(c_i = 1 \mid x_i) p(d_i \mid y_i = 1, x_i) \right] + \sum_{i \in I_0} \log \left( \sum_{c_i \in \{0, 1\}} p(y_i = 0 \mid x_i, c_i, e_i) p(c_i \mid x_i) \right) $$


위 로그를 취한 목적함수를 변형합니다. 미전환 유저들의 진짜 의도($c_i$)에 대한 임의의 가상 확률 분포 $q_i(c_i)$를 시그마 식 안에 추가합니다. 식의 원래 값을 변하게 하지 않기 위해 $q_i(c_i)$를 곱하고 동시에 나누어 줍니다. 이제 젠센의 부등식을 적용하여 로그 기호를 시그마 안쪽으로 밀어 넣습니다. $I_1$ 항은 '불확실성'이 전혀 없는, 정답이 이미 100% 공개된 상태이기 때문에 가상 확률 분포를 추가할 필요가 없기에 그대로 둡니다.

$$
\sum_{i \in I_0} \log \left( \sum_{c_i \in \{0, 1\}} q_i(c_i) \frac{p(y_i = 0 \mid x_i, c_i, e_i) p(c_i \mid x_i)}{q_i(c_i)} \right) \\[3pt]
\ge \sum_{i \in I_0} \sum_{c_i \in \{0, 1\}} q_i(c_i) \log \left( \frac{p(y_i = 0 \mid x_i, c_i, e_i) p(c_i \mid x_i)}{q_i(c_i)} \right) \\[3pt]
= \sum_{i \in I_0} \sum_{c_i \in \{0, 1\}} q_i(c_i) \log \left[ p(y_i = 0 \mid x_i, c_i, e_i) p(c_i \mid x_i) \right] - \sum_{i \in I_0} \sum_{c_i \in \{0, 1\}} q_i(c_i) \log q_i(c_i)
$$

우측의 $-\sum q_i \log q_i$ 항은 정보이론에서 말하는 분포 $q$의 엔트로피(Entropy) 입니다.EM 알고리즘의 최적화 단계(M-step)에서, 이 가상 분포 $q_i(c_i)$는 이전 스텝에서 계산되어 고정된 상수($\bar{q}_{ic_i}$)로 취급됩니다. 따라서 미분하면 0이 되어 사라지게 되므로 생략합니다. 정리하면 최종적으로 아래 등식이 성립합니다.

$$ \log p(D; \Theta) \ge Q(\Theta; \bar{\Theta}) \\[3pt]
= \sum_{i \in I_1} \log \left[ p(c_i = 1 \mid x_i) p(d_i \mid x_i, c_i = 1) \right] + \sum_{i \in I_0} \sum_{c_i \in \{0, 1\}} \bar{q}_{i c_i} \log \left[ p(y_i \mid x_i, c_i, e_i) p(c_i \mid x_i) \right] $$

EM(Expectation-Maximization) 알고리즘을 사용해 $\bar{q}_{ic}$와 실제 가중치 $\{w, V\}$를 시소 타듯이 번갈아 가며 구합니다.
```
[초기화] 가중치 w와 V를 아무 무작위 숫자로 설정한다.
  ↓
▶ [E-Step]
  1. 현재 세팅된 w와 V를 위의 1번, 2번 식에 대입한다.
  2. 그 대입해서 나온 확률값들을 곱해 미전환 유저들의 임시 사후 확률 값 "q_ic"를 계산해 고정한다.
  (사후 확률인 $\bar{q}_{ic}$는  Joint Model(전환 모델 $\times$ 시간 지연 모델)로 계산)
  ↓
▶ [M-Step]
  1. 위에서 계산해 둔 "q_ic"를 변하지 않는 고정 상수로 취급한다.
  2. 이 q_ic를 가중치 삼아 목적 함수 Q를 조립하고, L-BFGS 최적화기를 돌려 진짜 가중치인 w와 V를 더 나은 값으로 갱신한다.
  ↓
(이 과정을 w와 V가 더 이상 변하지 않고 수렴할 때까지 무한 반복한다!)
```
### Prediction
광고주나 시스템이 알고자 하는 목적에 따라 예측 방식을 이원화하여 처리합니다.  
- 시간 경과와 무관한 궁극적인 전환 확률 예측: 전환모델 ($p(c_i = 1 \mid x_i; w) = \left( 1 + \exp(-w^T x_i) \right)^{-1}$)
- 특정 시간 제한 $E$ 내에 "실제 전환이 완료될 확률" 예측: Joint Model 클릭 직후($0$)부터 마감 시간($E$) 사이의 모든 타이밍에 구매가 발생할 확률을 전부 더함(=적분) ($ \int_{0}^{E} p(c = 1 \mid x; w) \times p(d = t \mid x, c = 1; V) dt $)  

https://arxiv.org/abs/1802.00255

## Addressing Delayed Feedback for Continuous Training
### 지연된 피드백 딜레마
사용자가 광고 노출을 보고 난 후 실제 클릭을 하기까지 대기 시간이 발생한다는 점입니다. 최신 데이터로 실시간 학습을 하려 할 때, 아직 클릭 여부가 결정되지 않은 최근의 광고 노출을 어떻게 레이블링해야 하는지에 대한 기술적 딜레마가 존재합니다.
- 방안 A: 데이터가 수집될 때까지 일정 시간(Attribution Window) 대기 후 학습하면 데이터를 충분히 수집하기 위해 대기하는 동안 예측 모델이 노후화 되는 단점이 있습니다. 분석 결과에 따르면, Twitter의 경우 모델 업데이트가 단 5분만 지연되어도 서빙 성능에 치명적인 손상을 입는 것으로 밝혀졌습니다.
- 방안 B: 즉각적으로 '부정(Negative, 미클릭)' 레이블로 처리하여 즉시 학습하면 실제 데이터 분포보다 거짓 부정(Fake Negative) 샘플이 많아져 모델이 CTR을 과소평가하는 문제가 생깁니다.

### 제안: 얕은 선형 모델 기반이 아닌 딥 신경망 구조로 지연된 피드백 문제를 극복
본 연구는 광고가 노출된 시점부터 사용자의 실제 반응이 올 때까지 이추적하여 반영하겠다는 연속 학습 구조를 채택합니다. 때문에 학습에 쓰는 데이터는 "긍정(Positive) 레이블은 100% 진짜 클릭이지만, 부정(Negative) 레이블 속에는 진짜 미클릭과 아직 클릭이 지연된 가짜 부정(Fake Negative)이 마구 뒤섞여 있는 상태"가 됩니다.

### 모델 아키텍쳐
본 연구에서 지연된 피드백 문제를 해결하기 위해 평가 및 비교 분석한 두 가지 핵심 모델 구조의 상세 스펙을 설명하고 있습니다.

#### Logistic Regression 
이 모델이 예측하는 클릭률(CTR) $f_\theta(\mathbf{x})$는 다음과 같은 로지스틱 수식으로 계산됩니다:
$$
f_\theta(\mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{w}_c \cdot \mathbf{x})} = \sigma(\mathbf{w}_c \cdot \mathbf{x})
$$
- $\mathbf{x}$: 모델의 입력 벡터입니다. 특정 광고 요청 시점에 수집된 사용자(user) 정보 및 노출 후보 광고(ad candidates)들에 관련된 수천 개의 특성 피처들이 고차원의 희소 벡터(sparse representation) 형태로 표현된 것입니다.
- $\mathbf{w}_c$: 예측 클릭률(pCTR) 계산을 위해 학습 대상이 되는 가중치(weight) 벡터입니다.
- $\mathbf{w}_c \cdot \mathbf{x}$ : 가중치 벡터와 입력 피처 벡터 간의 내적(dot product) 연산입니다. 선형 결합 결과를 도출합니다.
- $\sigma(\cdot)$ 시그모이드 함수로 선형 내적 연산 결과를 입력받아 항상 0~1 사이의 값으로 바꾸어 예측 확률 값(pCTR)을 최종 계산해 냅니다.

#### Wide-and-Deep Model 
현대 추천 시스템에서 다뤄지는 대규모 고차원 희소 피처들의 복잡성과 다양성을 함께 해결하기 위해 고안된 하이브리드 신경망 구조입니다. 크게 두 종류의 예측 컴포넌트가 결합되어 동작합니다:
- Wide 영역 (일반화된 선형 모델): 원본 피처와 함께, 피처들을 임의로 결합한 교차곱 피처 변환(cross-product transformations)을 처리하여 모델에 비선형 학습 성능과 강력한 암기(Memorization) 능력을 주입합니다.
- Deep 영역 (피드포워드 신경망): 고차원의 희소한 카테고리형 피처들을 조밀하고 밀집된 형태의 저차원 임베딩 벡터(dense, low-dimensional representation)로 압축한 뒤 심층 레이어들을 통과시켜 복잡한 피처 조합을 일반화(Generalization)하여 깊이 학습합니다.

이 모델이 통합적으로 산출해 내는 클릭률 예측치 $f_\theta(\mathbf{x})$ 는 다음 수식으로 정의됩니다/
$$
f_\theta(\mathbf{x}) = \sigma(\mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T \alpha(l_f) + b)
$$
- $\mathbf{x}, \phi(\mathbf{x})$ : Wide 영역의 입력으로 연결(concatenation)된 벡터입니다. 원본 입력 피처 $\mathbf{x}$와 모델에 비선형 요소를 가미하기 위해 설계한 피처 간 교차곱 변환 변수 $\phi(\mathbf{x})$를 서로 결합한 것입니다.
- $\mathbf{w}_{wide}^T$ : Wide 구성 요소의 최종 학습 가중치 벡터입니다.
- $\alpha(l_f)$ : Deep 영역의 맨 마지막 레이어($l_f$)를 통과하며 최종적으로 출력되는 신경망의 활성화(activation) 값 벡터입니다.
- $\mathbf{w}_{deep}^T$: Deep 신경망 출력값 $\alpha(l_f)$ 에 최종적으로 곱해지는 딥러닝 출력 가중치 벡터입니다.
- $b$: 예측치 보정을 위해 더해지는 편향(bias) 파라미터입니다.
- $\sigma(\cdot)$ : Wide 파트의 선형 변환 결과와 Deep 파트의 마지막 신경망 연산 결과, 그리고 편향 파라미터까지 전부 합산한 최종 변환 값을 취합하여 0~1 사이의 실제 클릭률 범위로 축소하는 시그모이드 활성화 함수입니다.

### Loss Functions
논문에서 정의한 4가지 손실 함수를 비교 분석합니다.

#### Delayed Feedback Loss (지연 피드백 손실)
이 손실 함수는 사용자가 광고 노출을 본 후 클릭하기까지 걸리는 지연 시간 분포를 지수 분포(Exponential Distribution)로 가정하고, 이를 예측 모델과 함께 공동으로 최적화하는 기법입니다.

지연 시간 모델의 파라미터를 $\mathbf{w}_d$, pCTR 모델의 파라미터를 $\theta$ 라고 할 때, 정규화(Regularization) 파라미터 $\alpha$ 를 반영한 최종 목적 함수는 다음과 같습니다.
$$
\arg \min_{\theta, \mathbf{w}_d} L_{DF}(\theta, \mathbf{w}_d) + \alpha \left( \|\theta\|_2^2 + \|\mathbf{w}_d\|_2^2 \right), \\[3pt]
L_{DF}(\theta, \mathbf{w}_d) = - \sum_{\mathbf{x}, y} \log f_\theta(\mathbf{x}) - \sum_{\mathbf{x}, y=1} \left( \mathbf{w}_d \cdot \mathbf{x} - \lambda(\mathbf{x})d \right) - \sum_{\mathbf{x}, y=0} \log \left[ \exp(-f_\theta(\mathbf{x})) + \exp(-\lambda(\mathbf{x})e) \right], \\[3pt]
\lambda(\mathbf{x}) = \exp(\mathbf{w}_d \cdot \mathbf{x})
$$

- $f_\theta(\mathbf{x})$ : pCTR 예측 모델이 최종적으로 출력한 클릭 확률 값입니다.
- $\lambda(\mathbf{x})$ : 지연 시간 분포(지수 분포)의 파라미터로, 입력 피처 $\mathbf{x}$ 와 지연 모델의 가중치 $\mathbf{w}_d$ 의 선형 결합에 지수 함수를 취해 항상 양수가 되도록 정의합니다.
- $d$: 긍정 샘플($y=1$, 실제 클릭 발생)에 대해 기록된 '노출 후 클릭까지 걸린 시간(Time-to-click)'입니다.
- $e$: 부정 샘플($y=0$)에 대해 기록된 '노출 후 데이터 수집 스냅샷 시점까지 경과한 시간(Time elapsed)'입니다.
- 첫 번째 합산 항 (Positive 샘플 처리): 사용자가 결국 광고를 클릭한 경우($y=1$)에 해당하며, 클릭할 확률($f_\theta(\mathbf{x}$))과 특정 지연 시간 ($d$) 뒤에 행동을 보일 확률 밀도($\lambda(\mathbf{x}) \exp(-\lambda(\mathbf{x})d)$)를 동시에 극대화하도록 유도합니다.
- 두 번째 합산 항 (Negative/Unlabeled 샘플 처리): 아직 클릭이 관측되지 않은 경우($y=0$)에 해당합니다. 이 상태는 사용자가 실제로 광고를 영원히 클릭하지 않을 확률($1 - f_\theta(\mathbf{x})$)과, 클릭은 하겠지만 지연 시간 ($d$)가 현재 경과 시간 ($e$)보다 길어서 아직 관측되지 못했을 확률($f_\theta(\mathbf{x}) \exp(-\lambda(\mathbf{x})e)$)의 합으로 표현됩니다.

#### Positive-Unlabeled Loss (PU 손실)
PU 학습 관점을 도입하여, 관측된 편향 데이터셋의 모든 부정(Negative) 샘플을 단순히 '미분류(Unlabeled)' 샘플로 취급하고 학습을 전개하는 방식입니다. 
$$
L_{PU}(\theta) = - \sum_{\mathbf{x}, y=1} \left[ \log f_\theta(\mathbf{x}) - \log(1 - f_\theta(\mathbf{x})) \right] - \sum_{\mathbf{x}, y=0} \log(1 - f_\theta(\mathbf{x}))
$$
- $-\sum_{\mathbf{x}, y=0} \log(1 - f_\theta(\mathbf{x}))$: 모든 샘플(긍정/부정 모두 포함)을 우선 일차적으로 부정 샘플로 보고 일반적인 Log Loss를 적용해 업데이트하는 항입니다.
- $-\sum_{\mathbf{x}, y=1} \left[ \log f_\theta(\mathbf{x}) - \log(1 - f_\theta(\mathbf{x})) \right]$: 만약 어떤 샘플이 사후에 클릭되어 긍정($y=1$) 피드백이 들어오면 실행되는 교정 항입니다. 

실시간 스트리밍 도중 긍정 샘플이 확인되었을 때, 이 샘플은 과거(노출 직후)에 이미 '부정'으로 분류되어 $-\log(1 - f_\theta(\mathbf{x}))$ 만큼 모델 가중치를 부정 방향으로 잘못 업데이트했던 이력이 있습니다. 따라서 긍정 신호가 감지되는 순간, 과거의 잘못된 업데이트를 취소($- \log(1 - f_\theta(\mathbf{x})$ 제거)하고, 동시에 올바른 긍정 방향의 업데이트($\log f_\theta(\mathbf{x})$)를 한 번에 수행해 가중치를 제자리로 돌려놓는 구조입니다.

### Fake Negative Weighted 
중요도 샘플링(Importance Sampling) 이론을 기반으로 설계되었으며, 관측된 편향 분포 $b(\mathbf{x}, y)$ 하에서의 기댓값 계산을 실제 데이터 분포 $p(\mathbf{x}, y)$ 기준으로 무편향(Unbiased)이 되도록 샘플 가중치를 직접 조절하는 손실 함수입니다.  
이때, 편향된 관측 분포와 실제 분포 사이의 가중치를 계산하기 위해 다음과 같은 가설을 전제합니다.
1. $b(\mathbf{x} | y = 0) = p(\mathbf{x})$: 부정으로 관측된 샘플들의 피처 분포는 전체 입력 피처 분포와 동일함.
2. $b(\mathbf{x} | y = 1) = p(\mathbf{x} | y = 1)$: 긍정으로 관측된 샘플들의 피처 분포는 실제 진짜 긍정 샘플들의 피처 분포와 완벽히 일치함.
3. $b(y = 1) = \frac{p(y = 1)}{1 + p(y = 1)}$: 모든 샘플이 최초에 부정 레이블로 인입된 후, 클릭 시점에 긍정 레이블로 중복 인입되므로 전체 데이터 모수가 $1+p(y=1)$ 배로 늘어남을 반영한 사전 확률.

위 가설들을 바탕으로 베이즈 정리(Bayes' Theorem)를 적용해 유도한 '관측 데이터에서 긍정 및 부정으로 분류될 확률'은 다음과 같습니다.

$$ b(y = 1 | \mathbf{x}) = \frac{p(y = 1 | \mathbf{x})}{1 + p(y = 1 | \mathbf{x})} $$

$$ b(y = 0 | \mathbf{x}) = 1 - b(y = 1 | \mathbf{x}) = \frac{1}{1 + p(y = 1 | \mathbf{x})}  $$

이 식들을 원래 분포의 크로스 엔트로피 식에 대입하여 최종 도출해 낸 중요도 샘플링 손실 함수 $L_{IS}(\theta)$ 는 다음과 같습니다.
$$ 
L_{IS}(\theta) = - \sum_{\mathbf{x}, y} \left[ b(y = 1 | \mathbf{x}) (1 + p(y = 1 | \mathbf{x})) \log f_\theta(\mathbf{x})+ b(y = 0 | \mathbf{x}) (1 - p(y = 1 | \mathbf{x})) (1 + p(y = 1 | \mathbf{x})) \log (1 - f_\theta(\mathbf{x})) \right] 
$$

실제 환경에서는 참 분포 확률 $p(y=1|\mathbf{x})$ 를 직접 알 수 없으므로, 이를 모델의 현재 예측치인 $f_\theta(\mathbf{x})$ 로 대체하여 아래와 같은 최종 가중치를 샘플에 부여합니다.
- 긍정 샘플($y=1$)에 적용할 가중치: $1 + f_\theta(\mathbf{x})$
- *부정 샘플($y=0$)에 적용할 가중치: $(1 - f_\theta(\mathbf{x}))(1 + f_\theta(\mathbf{x}))$

이 손실 함수의 손실 값에 대한 예측치 \\(f_\theta\\)의 미분(Gradient)은 다음과 같이 유도됩니다.
$$
\frac{\partial L_{IS}}{\partial f_\theta} = \frac{(1 + f_\theta(\mathbf{x}))(f_\theta(\mathbf{x}) - p(y=1 | \mathbf{x}))}{(1 + p(y=1 | \mathbf{x})) f_\theta(\mathbf{x})}
$$
* 위 그래디언트 식(Equation 11)을 보면 0 이 되는 유일한 지점은 $f_\theta(\mathbf{x}) = p(y=1 | \mathbf{x})$ 일 때입니다. 즉, 모델의 예측값 $f_\theta(\mathbf{x})$ 가 실제 그라운드 트루스 확률 $p(y=1|\mathbf{x})$ 에 도달했을 때 그래디언트가 정확히 0 이 되어 최적 수렴하게 되며, 수렴 과정 중에도 그래디언트가 언제나 올바른 방향을 가리키게 됨을 수학적으로 보장합니다.

### Fake Negative Calibration (거짓 부정 보정)
손실 함수 자체를 복잡하게 변경하는 대신, 인프라 비용을 최소화하기 위해 고안된 매우 실용적인 2단계 접근법입니다.

1. 편향이 포함된 스트리밍 데이터를 그대로 사용하여 일반적인 로그 손실(Log loss) 함수로 모델을 우선 학습시킵니다. 이 모델이 예측하는 값은 왜곡된 편향 분포인 $b(y=1|\mathbf{x})$ 를 따르게 됩니다.
2. 학습이 완료된 후, 출력되는 예측값에 대해 $b(y = 1|x) = \frac{p(y = 1|x)}{1 + p(y = 1|x)}$ 을 역으로 풀어내어 실제 물리적인 클릭 확률인 $p(y=1|\mathbf{x})$로 사후 매핑(Calibration)을 수행합니다.

$$
p(y = 1 | \mathbf{x}) = \frac{b(y = 1 | \mathbf{x})}{1 - b(y = 1 | \mathbf{x})}
$$

관측 데이터셋 내에서는 모든 진짜 긍정 샘플에 대응하여 처음에 임시 수집된 거짓 부정(FN) 샘플이 1:1 비율로 쌍을 이루어 존재합니다. 이 때문에 편향된 분포에서의 긍정 확률 $b(y=1|\mathbf{x})$ 는 이론적으로 절대 0.5 를 넘을 수 없습니다. ($b(y=1|\mathbf{x}) \le 0.5$)  
따라서 분모인 $1 - b(y=1|\mathbf{x})$ 는 항상 0.5 이상이 보장되므로, 최종 보정된 실제 확률 값 $p(y=1|\mathbf{x})$ 는 언제나 0~1 사이의 적법한 확률 분포 범위 내에 머물게 됩니다.

### Conclusion
Wide & Deep 모델이 모든 손실 함수 영역에서 로지스틱 회귀 모델보다 압도적으로 뛰어난 기본 체력을 보여주었습니다.
특히 Wide & Deep 모델 기반에서 FN Calibration(RCE 13.58), PU Loss(RCE 13.57), FN Weighted(RCE 13.54) 삼총사가 오프라인 실험에서 가장 높은 퍼포먼스를 보여주었습니다. 즉, 모델이 복잡해질수록(딥러닝) 단순 지연 피드백 보정보다 중요도 샘플링과 같은 정교한 확률 보정이 더 효과적임을 시사합니다. 
대규모 스트리밍 데이터 기반의 연속 학습 환경에서 지연 피드백 바이어스를 교정하는 데 Wide & Deep 모델과 FN Weighted / FN Calibration 손실 함수의 조합이 가장 최적의 솔루션임을 증명했습니다.

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