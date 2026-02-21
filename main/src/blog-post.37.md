# Diffusion Model
## Deep Unsupervised Learning using Nonequilibrium Thermodynamics
https://arxiv.org/pdf/1503.03585

## Denoising Diffusion Probabilistic Models
https://arxiv.org/abs/2006.11239

## Preliminary
### Latent variable model
Latent, or “hidden,” variable modeling is a statistical method that studies hidden concepts by analyzing measurable indicators that reflect them. To estimate these hidden constructs, researchers begin with observable data, like test scores or behaviors, and use them to make inferences about underlying concepts, including academic ability, mental health, or customer satisfaction. 

A latent variable model is a statistical model that relates a set of observable variables (also called manifest variables or indicators) to a set of latent variables. It is assumed that the responses on the indicators or manifest variables are the result of an individual's position on the latent variable(s), and that the manifest variables have nothing in common after controlling for the latent variable (local independence).

Different types of latent variable models can be grouped according to whether the manifest and latent variables are categorical or continuous.

| **Manifest Variables (관측변수)** | **Latent Variables: Continuous (연속형)** | **Latent Variables: Categorical (범주형)** |
| ----------------------------- | -------------------------------------- | --------------------------------------- |
| **Continuous (연속형)**          | Factor Analysis (요인분석)                 | Item Response Theory (문항반응이론)           |
| **Categorical (범주형)**         | Latent Profile Analysis (잠재프로파일분석)     | Latent Class Analysis (잠재계층분석)          |

### Markov Chain
A Markov chain is a way to describe a system that moves between different situations called "states", where the chain assumes the probability of being in a particular state at the next step depends solely on the current state. 

For example, let's consider two-state Markov chain below.  
<img src="images/blog0_markov_chain.png" alt="Markov Chain" width="150"/>     

If in state A:
- Stays in A: probability 0.6
- Moves to E: probability 0.4  

If in state E:
- Moves to A: probability 0.7
- Stays in E: probability 0.3

Also note that a Markov chain can be illustrated as a directed graph, where nodes represent the states (A, E), arrows indicate possible transitions and the numbers on arrows show transition probabilities.

### Mathematical Background of Diffusion Model
확산 모델(Diffusion Models)의 수학적 기초를 설명합니다. 확산 모델은 Latent variable model로 아래의 적분 형태로 나타낼 수 있습니다.
$$p_\theta(x_0) := \int p_\theta(x_{0:T}) dx_{1:T}$$
- $x_0$: 우리가 실제로 관찰하는 데이터(예: 이미지)입니다.
- $x_1, \dots, x_T$: "잠재 변수(Latents)"라고 불리며, 데이터가 생성되는 과정 중에 존재하는 중간 단계들을 의미합니다.
- $p_\theta(x_0)$: 모델이 최종적으로 생성해낸 데이터 $x_0$의 확률 분포입니다. 이 분포가 실제 데이터 분포인 $q(x_0)$와 최대한 일치하도록 만드는 것이 모델의 목표입니다.

이 수식을 적분으로 나타낼 수 있는 이유는 아래 2가지 입니다.
1. 결합 확률 분포($p_\theta(x_{0:T})$): 모델은 노이즈($x_T$)부터 시작해 여러 단계($x_{T-1}, \dots, x_1$)를 거쳐 데이터($x_0$)를 생성하는 전체 경로에 대한 확률을 가집니다. 이를 Reverse Process이라고 부릅니다.
2. 주변 확률 계산: 우리가 보고 싶은 것은 오직 최종 결과물인 $x_0$뿐입니다. 따라서 모든 가능한 중간 경로들($x_{1:T}$)에 대해 확률을 다 더해주는(연속 변수이므로 적분하는) 과정을 통해 $x_0$만이 나타날 확률을 구하는 것입니다.

확산 모델은 데이터를 생성하는 Reverse Process과 데이터를 파괴하는 Forward Process로 이루어져 있습니다.

#### Reverse Process
완전한 노이즈($x_T$)에서 시작하여, 신경망($\theta$)이 예측하는 가우시안 분포를 따라 한 단계씩($T \rightarrow T-1 \rightarrow \dots \rightarrow 0$) 노이즈를 걷어내면(Denoising), 최종적으로 실제 데이터($x_0$)를 얻을 수 있다는 과정을 수학적으로 정리합니다.

$$
p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) \\[5pt]
p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I}), \quad 
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
$$

- $p_\theta(\mathbf{x}_{0:T})$: 이 모델이 생성하는 전체 경로($x_0, x_1, \dots, x_T$)에 대한 결합 확률 분포(Joint Distribution).
- $\theta$ (아래 첨자): 딥러닝 모델(신경망)의 학습 가능한 Parameters 를 나타냅니다. 즉, 이 분포는 고정된 것이 아니라 데이터를 통해 학습되는 것임을 의미.
- $p(\mathbf{x}_T)$: Reverse Process의 시작점인 Prior Distribution 입니다. 아무런 정보가 없는 완전한 노이즈 상태의 가우시안 분포입니다.
- $\prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$: $t=1$부터 $T$까지의 모든 단계에 대한 확률을 다 곱한다는 의미입니다. 이는 과정이 Markov Chain임을 나타내며, 현재 단계($x_t$)는 바로 다음 단계($x_{t-1}$)에만 영향을 줍니다.
- $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$: Reverse Process Transition으로 노이즈가 더 많은 상태인 $\mathbf{x}_t$가 주어졌을 때, 노이즈가 조금 제거된 이전 상태인 $\mathbf{x}_{t-1}$이 어떠할지를 예측하는 확률 모델입니다.
- $\mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$: 다음 단계를 현재 단계의 평균과 분산 기반의 가우시안 분포로 예측

#### Forward Process
순방향 과정(Forward Process) 또는 확산 과정(Diffusion Process)을 정의합니다.  
이 과정은 데이터($\mathbf{x}_0$)에서 시작하여, 미리 정해진 스케줄($\beta_t$)에 따라 아주 조금씩 가우시안 노이즈를 섞어가는 과정을 통해 데이터의 정보를 파괴해 나가는 과정입니다. 이 과정을 $T$번(예: 1000번) 반복하면, 원래 데이터의 형체는 완전히 사라지고 표준 가우시안 노이즈($\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$)에 가까운 상태가 됩니다.

$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) := \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1}), \quad q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

- $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$: 실제 데이터 $\mathbf{x}_0$가 주어졌을 때, 시점 $1$부터 $T$까지 생성되는 모든 중간 단계들의 조건부 결합 확률 분포(Joint Distribution)입니다. 단, 역과정($p_\theta$)과 달리, 고정된 과정으로 신경망이 학습하는 대상이 아니라 미리 정해진 규칙에 따라 노이즈를 더하는 과정입니다.
- $\prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})$: $t=1$부터 $T$까지의 모든 단계에 대한 확률을 다 곱한다는 의미입니다. 이는 과정이 Markov Chain임을 나타내며, 현재 단계($x_t$)는 바로 다음 단계($x_{t-1}$)에만 영향을 줍니다.
- $\mathcal{N}(\dots)$: 각 단계에서 추가되는 노이즈가 가우시안 분포를 따른다는 것을 의미합니다.
- $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$: 가우시안 분포의 평균은 이전 단계의 데이터에 $\sqrt{1 - \beta_t}$를 곱하여 크기를 약간 줄입니다. 왜냐하면 단순히 노이즈를 더하기만 하면 데이터의 전체 분산이 계속 커지기 때문입니다. 이 스케일링 인자를 통해 노이즈를 더한 후에도  전체 데이터의 분산(Sclae)이 일정하게 유지되도록 조절합니다.
- $\beta_t \mathbf{I}$: 각 시점 $t$에서 추가되는 노이즈의 양입니다.
- $\beta_1, \dots, \beta_T$: 분산 스케줄(Variance Schedule)로  소스의 실험에서는 $10^{-4}$에서 $0.02$까지 선형적으로 증가하는 값을 사용했습니다.
- $\mathbf{I}$: 항등 행렬로, 모든 차원에 독립적으로 동일한 양의 노이즈가 추가됨을 의미합니다.


## High-Resolution Image Synthesis with Latent Diffusion Models
https://arxiv.org/pdf/2112.10752

## DENOISING DIFFUSION IMPLICIT MODELS
https://arxiv.org/pdf/2010.02502
