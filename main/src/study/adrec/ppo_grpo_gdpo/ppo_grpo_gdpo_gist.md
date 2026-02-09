# 0. Preliminary
**$\pi_{\theta}(a_t|s_t)$ - 정책 함수**  
상태 $s_t$가 주어졌을 때 행동 $a_t$ 를 선택할 조건부 확률 분포 → 모델이 현재 상황($s_t$)에서 어떤 행동($a_t$)을 취할 가능성이 얼마나 높은지를 수치화합니다.

**$A^\pi(s_t, a_t)$ - 어드밴티지 함수**  
정책 $\pi$ 에 따른 어드밴티지 함수로 특정 행동 $a_t$가 정책의 평균적인 행동보다 얼마나 더 좋은지를 나타냅니다.

$$
A^\pi(s_t, a_t) := Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

- $Q^\pi(s_t, a_t)$: 상태 $s_t$ 에서 특정 행동 $a_t$ 를 취한 다음, 그 이후로 부터는 정책 $\pi$를 따랐을 때의 총 보상의 합. = State-Value Function
- $V^\pi(s_t)$: 상태 $s_t$ 에서 단순히 현재 정책 $\pi$ 를 따랐을 때 기대되는 총 보상의 합. = Action-Value Function
- $A^\pi(s_t, a_t)$: 내가 지금 한 행동($a_t$) 이 평소 하던 대로($\pi$) 했을 때보다 얼마나 더 좋은가?를 측정하는 것.

# 1. 정책 경사법 (Policy Gradient Method) 이란?
정책 최적화 문제에서는 매 타임스텝마다 보상 $r_t$ 를 받으며, 목표는 기대 총 보상의 합을 최대화하는 것입니다.

$$
\text{Maximize } \mathbb{E}\left[\sum_{t=0}^{\infty} r_t\right]
$$

"Policy Gradient methods" 를 사용하면 위 "the expected total reward"를 최대화(maximize) 하기 위해 반복적으로 gradient를 추정합니다. 이 "Policy Gradient" 는 일반적으로 아래와 같은 form을 가집니다. 모델은 아래 식을 통해 보상이 높았던 행동의 발생 확률을 높이도록 스스로를 교정합니다.

$$
g = \mathbb{E}\left[\sum_{t=0}^{\infty} \Psi_t \nabla_\theta \log \pi_\theta (a_t \mid s_t)\right]
$$

- $g$(Policy Gradient): 기대 총 보상($\mathbb{E}\left[\sum_{t=0}^{\infty} r_t\right]$)을 최대화하기 위해 정책 파라미터 $\theta$ 가 이동해야 할 경사(Gradient) 방향입니다.
- $\mathbb{E}$: 정책에 의해 생성된 모든 가능한 궤적(Trajectory)에 대한 평균값을 의미합니다.
- $\nabla_\theta \log \pi_\theta (a_t \mid s_t)$: 상태 $s_t$ 에서 행동 $a_t$ 를 선택할 로그 확률의 기울기입니다. 이 항은 파라미터 $\theta$ 를 수정하여 해당 행동이 일어날 확률을 높이는 방향을 가리킵니다.
- $\Psi_t$(Scalar Weight): 해당 타임스텝의 행동이 "얼마나 좋았는지"를 나타내는 가중치입니다. 이 값이 그래디언트의 방향을 결정하는 지표가 됩니다.

GAE 에서는 $\Psi_t$로 어드밴티지 함수를 사용합니다. 어드밴티지 함수는 특정 행동 $a_t$가 정책의 평균적인 행동보다 얼마나 더 좋은지를 나타냅니다.

하지만 실제 계산에서 무한한 타임스텝의 보상을 합산하면 분산이 매우 커집니다. 따라서 이를 해결하기 위해 분산 감소를 위한 할인 계수($\gamma$) 를 도입합니다.

$$
g^\gamma := \mathbb{E}_{\substack{s_{0:\infty} \\ a_{0:\infty}}} \left[ \sum_{t=0}^\infty A^{\pi,\gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right], \quad A^{\pi,\gamma}(\mathbf{s}_t, \mathbf{a}_t) := Q^{\pi,\gamma}(\mathbf{s}_t, \mathbf{a}_t) - V^{\pi,\gamma}(\mathbf{s}_t).
$$

강화학습에서는 보상을 높여야 하므로 위에서 예측한 기울기(Gradient) 방향으로 매개변수를 더해주는 Gradient Ascent 방법을 사용합니다.

$$
\theta_{i+1} \leftarrow \theta_i + \alpha_i g_i
$$

- $\theta_i$: 현재 단계($i$)의 정책함수의 파라미터입니다.
- $\theta_{i+1}$: 업데이트된 다음 단계의 파라미터입니다.
- $\alpha_i$: 학습률(Learning rate)로, 한 번의 업데이트에서 정책을 얼마나 변경할지 결정합니다.
- **$g_i$**: 파라미터 공간에서 추정된 기울기로 높은 보상과 연결된 행동의 발생 확률을 높이는 방향을 가리킵니다.

# 2. PPO (Proximal Policy Optimization)
## PPO 알고리즘
PPO도 Policy Gradient Method 의 한 종류로 Gradient Ascent을 통해 미분 가능한 정책 함수($\pi$)에 의해 action이 결정되는 강화학습(RL) 에이전트를 훈련하는 데 사용됩니다. 일반적인 Policy Gradient Method는 정책 업데이트($\theta_{i+1} \leftarrow \theta_i + \alpha_i g_i$) 단계를 거쳐 에이전트가 기대 보상을 점점 더 높이는 방향으로 학습하지만 이런 일반적인 Policy Gradient method는 불안정할 수 있습니다. 즉, 보폭(step size = learning rate)이 너무 크면 정책이 최적이지 않은 방향으로 미끄러지며(drift) 최적화로의 복구가 거의 불가능해집니다. 그렇지만 반대로 보폭이 너무 작으면 전반적인 훈련 효율성이 떨어집니다.

이러한 불안정성을 해결하기 위해, PPO는 에이전트의 정책 업데이트가 한 번의 step에 너무 커지지 않도록 제한하는 "clip 함수"를 도입합니다. 이를 통해 Gradient Ascent 과정에서 미끄러짐(drift)에 대한 위험 없이 더 큰 보폭을 사용할 수 있습니다. PPO의 손실 함수는 다음과 같이 정의됩니다.

$$
\mathcal{J}_{\text{PPO}}(\theta)
= \mathbb{E}\!\left[
\frac{1}{|o|} \sum_{t=1}^{|o|}
\min\!\Big(
\frac{\pi_\theta(o_t \mid q, o_{\lt t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{\lt t})} A_t,\,
\text{clip}\big(\frac{\pi_\theta(o_t \mid q, o_{\lt t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{\lt t})}, 1 - \varepsilon, 1 + \varepsilon\big) A_t
\Big)
\right]
$$

- $\pi_\theta, \pi_{\theta_{\text{old}}}$: 신규 및 이전 정책 모델입니다.
- $\pi_\theta(o_t \mid q, o_{\lt t})$: 입력 프롬프트 $q$와 이전에 생성된 모든 토큰 $o_{\lt t}$ 가 주어졌을 때, 토큰 $o_t$를 생성할 확률입니다.
- $q, o$: 각각 질문 데이터셋과 이전 정책 $\pi_{\theta_{\text{old}}}$에서 샘플링된 질문과 출력입니다.
- $\varepsilon$: 훈련 안정화를 위해 PPO에서 도입된 클리핑 관련 하이퍼파라미터입니다.
- $A_t$: GAE(Generalized Advantage Estimation)를 적용하여 계산된 어드밴티지(advantage)입니다.
- **clip**
  - $\frac{\pi_\theta(o_t \mid q, o_{\lt t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{\lt t})}> 1 + \varepsilon$: 새 모델이 동일한 행동에 대해 너무 높은 확률을 부여할 때 → 이를 클리핑(제한)합니다. 즉, 만약 어떤 행동이 보상이 좋아서(Advantage > 0) 신규 정책($\pi_\theta(o_t \mid q, o_{\lt t})$) 이 이 행동을 할 확률을 계속 높이다가, 그 비율이 $1 + \varepsilon$ 을 넘어서게 되면 제한(clip)이 작동합니다.
  - $\frac{\pi_\theta(o_t \mid q, o_{\lt t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{\lt t})} < 1 - \varepsilon$: 너무 낮은 확률을 부여할 때 → 이 역시 클리핑합니다. 반대로 보상이 나빠서 확률을 낮추다가 비율이 $1 - \varepsilon$ 밑으로 떨어져도 마찬가지로 제한이 걸립니다.
  - Clip 함수가 기존 정책 대비 신규 정책의 확률 비율을 제한하는 이유는 모델의 파라미터($\theta$)가 너무 멀리 이동하는 것을 막기 위해서입니다.

PPO는 이처럼 clip 방식을 사용하여 새 정책을 이전 정책과 **"가깝게(close)"** 유지함으로써 파괴적인 대규모 업데이트를 방지합니다. 이것이 기존의 기본적인 정책 경사법과 비교하여 PPO의 훈련을 안정화하는 핵심 요소입니다.

보상 모델은 다음과 같이 정의됩니다.

$$
r_t = r_\phi(q, o_{\le t}) - \beta \log \frac{\pi_\theta(o_t \mid q, o_{\lt t})} {\pi_{\text{ref}}(o_t \mid q, o_{\lt t})}
$$

- $r_\phi$: 주어진 입력 $q$에 대해 모델의 출력 $o$가 얼마나 인간이 선호하는 것인지를 알려주는 스칼라 점수 $r_\phi$를 제공하는 학습된 보상 모델입니다. 이는 별도로 학습되므로 PPO 과정 중에는 고정(fixed)됩니다.
- $\pi_{\text{ref}}$: 학습의 기준이 되는 고정된 모델을 의미하며, 일반적으로 강화학습 이전 단계인 지도 미세 조정(SFT)을 마친 초기 모델입니다.

즉, 첫 번째 항인 $r_\phi(q, o_{\le t})$는 더 나은 답변을 생성하도록 장려하는 반면, 두 번째 항(KL-패널티 = KL-발산)은 원래 모델에서 급격하게 변화하는 것을 억제합니다. 이를 통해 학습 중인 모델이 레퍼런스 모델(원래 모델)에서 너무 멀리 벗어나지 않도록(drift) 억제하여 학습의 안정성을 높이고 보상 모델의 과적합을 방지합니다.

이제 어드밴티지는 GAE(Generalized Advantage Estimation)를 통해 일련의 $r_t$ 값으로부터 계산됩니다.

$$
A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k
\big( r_{t+k} + \gamma V_\psi(o_{t+k+1}) - V_\psi(o_{t+k}) \big), \quad
V_\psi(o_t) \approx \mathbb{E}[R_t \mid o_t]
$$

여기서 $R_t$는 토큰 $o_t$에서 시작하는 기대 미래 누적 보상입니다. $V_\psi(o_t)$는 가치 모델(value model)로 단순히 현재 정책 $\pi$ 를 따랐을 때 기대되는 총 보상의 합을 의미합니다. 이는 현재 상황이 이미 얼마나 좋은지에 대한 베이스라인(baseline) 추정치를 제공합니다.

결론적으로, **보상(reward) → 어드밴티지(advantage) → PPO 최적화**의 흐름으로 진행됩니다. 이해를 돕기 위해 관련 이미지를 함께 확인하시기 바랍니다.

> **Gist 사용 시:** 이미지는 절대 URL로 넣어야 합니다. (예: `![PPO vs GRPO](https://...이미지URL...)`)
>
> ![PPO vs GRPO](https://via.placeholder.com/500x300?text=PPO+vs+GRPO+%28%EC%9D%B4%EB%AF%B8%EC%A7%80+URL%EB%A5%BC+%EB%84%A3%EC%96%B4%EC%A3%BC%EC%84%B8%EC%9A%94%29)

## GAE에서 어드밴티지 함수 추정
실제 할인된 ($\gamma$ 가 적용된) 어드밴티지 함수 $A^\pi(s_t, a_t)$ 는 알 수 없으므로, 전체 궤적 데이터를 사용하여 계산한 근사치를 대신 사용해야 하므로 이를 추정하는 방법을 다룹니다. 즉, ground truth 할인된 어드밴티지 함수($A^\pi(s_t, a_t)$)를 정확하게 추정하는 $\hat{A}$ 를 만드는 것입니다.

어드밴티지 추정치 $\hat{A}$ 를 만들 때는 TD-Residual 개념을 사용합니다. 근사 가치 함수 $V$를 사용하여 다음과 같이 정의 할 수 있습니다.

$$
\delta_t^{V^{\pi,\gamma}} = r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)
$$

위 식은 현재 상태 $s_t$ 에서 행동 $a_t$ 를 취했을 때 얻은 즉각적인 보상($r_t$) 과 다음 상태의 가치($\gamma V^{\pi,\gamma}(s_{t+1})$)의 합에서, 원래 예상했던 현재 상태의 가치($V^{\pi,\gamma}(s_t)$)를 뺀 값으로 나타냅니다. 즉 TD-Residual은 행동 $a_t$ 에 대한 어드밴티지의 추정치로 간주할 수 있습니다.

이를 $k$ 타임스텝으로 확장하면,

$$
\hat{A}_t^{(1)} := \delta_t^V = -V(s_t) + r_t + \gamma V(s_{t+1})
$$

$$
\hat{A}_t^{(2)} := \delta_t^V + \gamma\delta_{t+1}^V = -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})
$$

$$
\hat{A}_t^{(3)} := \delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V = -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3})
$$

$$
\cdots
$$

$$
\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta^{V}_{t+l} = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
$$

$$
\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{k-1} \gamma^l r_{t+l}
$$

위를 바탕으로 일반화된 어드밴티지 추정치(Generalized Advantage Estimator, GAE)를 정의합니다. GAE는 현재 시점($1$)에서 $\infty$ 단계까지의 어드밴티지 추정치들을 적절한 비율($\lambda$)로 지수적으로 가중 평균한 것입니다.

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} := (1 - \lambda) (\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \ldots)
$$

$$
= (1 - \lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \ldots)
$$

$$
= (1 - \lambda)\left(\delta_t^V (1 + \lambda + \lambda^2 + \ldots ) + \gamma\delta_{t+1}^V (\lambda + \lambda^2 + \lambda^3 + \ldots ) + \gamma^2\delta_{t+2}^V (\lambda^2 + \lambda^3 + \lambda^4 + \ldots ) + \ldots \right)
$$

$$
= (1 - \lambda) \left( \delta_t^V \left( \frac{1}{1 - \lambda} \right) + \gamma\delta_{t+1}^V \left( \frac{\lambda}{1 - \lambda} \right) + \gamma^2\delta_{t+2}^V \left( \frac{\lambda^2}{1 - \lambda} \right) + \ldots \right) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V
$$

즉 GAE는 TD residual을 이용하여 여러 개의 k-스텝 어드밴티지(Q-V) 추정치를 $\lambda$ 를 이용해 지수 가중 평균(weighted sum)하는 개념입니다. $1 - \lambda$ 전체 가중치의 합을 1로 맞추기 위한 정규화 계수 입니다.

$$
\hat{g} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{\infty} \hat{A}_t^n \nabla_\theta \log \pi_\theta (a_t^n \mid s_t^n)
$$

구한 어드밴티지 추정치 $\hat{A}$를 사용하여 **각 개별 에피소드($n$)의 정책 경사를 계산한 후, 이를 배치 단위($N$)로 평균 내어 최종 정책 경사($\hat{g}$)를 구하게 됩니다.**

(PPO의 손실함수를 미분해서 나온 policy gradient 수식입니다.)

$$
\nabla_\theta L(\theta) = \mathbb{E}_{t} \left[ \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right] = \mathbb{E}_{t} \left[ \hat{A}_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] = \hat{g}
$$

### 3. GAE 에서 $\lambda$에 따른 편향-분산 트레이드오프
#### 3-1. 분산 (Variance): 샘플링의 불확실성
Policy Gradient Method에서 분산이 높다는 것은 수집된 데이터(샘플)에 따라 기울기 추정값이 너무 크게 변한다는 것을 의미합니다.
- 원인: 행동의 효과는 과거와 미래의 수많은 다른 행동들과 뒤섞여(confounded) 나타나기 때문에, 실제 궤적의 보상 합계를 직접 사용하면 타임 호라이즌이 길어질수록 분산이 불리하게 커집니다.
- 문제점: 분산이 높으면 학습에 많은 양의 샘플이 필요하게 되며, 학습 과정이 매우 불안정해집니다.
- 해결책: 가치 함수($V$)를 사용하여 미래의 수많은 무작위 경로에서 오는 '불확실성(분산)'을 미리 계산된 '안정적인 평균값'으로 대체함으로써 분산을 실질적으로 줄일 수 있습니다. ("실제 보상의 합" 은 실제 측정된 1개의 궤적이고 매 실행마다 크게 달라지는 높은 분산을 가지고 있으니 측정된 값을 쓰지 않고 가치함수로 미래 보상들의 기대 값(평균값)을 써서 분산을 줄임)

##### 실제 보상의 합 vs 가치함수
**Empirical Returns, $R_t$**

$$
R_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l}
$$

특정 환경에서 실제로 움직이며 얻은 개별 보상($r$)들의 단순 누적 합계로 확률적인 기댓값이 아니라, 한 번의 실행(Rollout)을 통해 얻은 실제 측정값입니다.

**Value Function, $V^\pi(s_t)$**

$$
V^\pi(s_t) = \mathbb{E}_{s_{t+1:\infty}, a_{t:\infty}} \left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \mid s_t \right]
$$

가능한 모든 미래 시나리오에 대한 통계적 기댓값으로 위에서 설명한 실제 보상의 합($R_t$)에 대해 기댓값을 취한 것과 같습니다. 기대값은 '값 $\times$ 그 값이 일어날 확률'을 모두 더하는 아래와 같은 식임을 참고하면 이해하기 쉽습니다.

$$
\mathbb{E} [f(a|s)] = \sum_{s} Pr(s) \sum_{a} \pi_\theta(a|s) f(a|s)
$$

- $\pi_\theta(a|s)$: 특정 상태에서 어떤 행동을 할 확률입니다.
- $Pr(s)$: 특정 상태에 도달할 확률입니다.
- $f(a|s)$: 그 행동을 했을 때 얻게 되는 보상입니다.

### 3-2. 편향 (Bias): 근사의 부정확성
편향은 추정값이 실제 정답(ground truth 어드밴티지)에서 체계적으로 벗어나는 정도를 말합니다.
- 원인: 실제 보상 값 대신 학습된 가치 함수($V$)를 사용하여 어드밴티지를 추정할 때, 가치 함수 자체가 정확하지 않으면 편향이 발생합니다.
- 문제점: 높은 분산은 더 많은 샘플로 해결할 수 있지만, 편향은 훨씬 더 치명적입니다. 샘플이 무한히 많아도 편향이 있으면 알고리즘이 수렴에 실패하거나 최적이 아닌 엉뚱한 결과로 수렴할 수 있습니다.

### 3-3. GAE의 절충 전략: $\gamma$와 $\lambda$
GAE는 두 가지 파라미터($\gamma, \lambda$)를 통해 분산과 편향의 균형을 조절합니다.

**$\gamma$ 의 역할**
- 분산 감소: 미래 보상의 가중치를 낮추어 지연된 효과로 인한 노이즈(분산)를 줄입니다.
- 편향 유발: $\gamma < 1$을 사용하면 가치 함수가 정확하더라도 정책 경사 추정치에 항상 편향이 도입됩니다. (가치 함수가 '할인된 세계'에서는 완벽할지 몰라도, '할인이 없는 원래의 세계'의 관점에서 보면 먼 미래의 보상을 무시하고 있는 셈이 되므로 정답에서 벗어난 편향(Bias)이 발생하는 것)

**$\lambda$ (GAE 파라미터)의 역할**  
$\lambda$는 여러 단계의 어드밴티지 추정치들을 어떻게 섞을지 결정하며, 가치 함수의 정확도에 따라 편향을 조절합니다.

##### $\lambda = 1$ (High Variance, Low Bias)
- $\hat{A}_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$ , 실제 보상의 합(Empirical returns)에서 베이스라인을 뺀 방식입니다.
- 실제 보상의 합을 그대로 사용하기 때문에 에피소드마다 결과가 크게 달라지는 높은 분산 문제를 겪습니다.
- 편향의 원인이 될 가치함수 $V(s_t)$가 단순히 전체 보상에서 일정한 값을 빼주는 '베이스라인(Baseline)' 역할만 수행하기 때문에 편향이 없습니다.

##### $\lambda = 0$ (Low Variance, High Bias)
- $\hat{A}_t = \delta_t^V$, 1단계 TD 잔차($\delta_t$)만을 사용합니다.
- 미래 시점의 TD 잔차($\delta_{t+l}$)들에 $(\gamma\lambda)^l$이라는 가중치를 주어 지수적으로 감쇠시킵니다. 이는 멀리 있는 불확실한 미래 보상들의 영향력을 줄여서 전체적인 추정치의 분산을 낮추는 효과를 줍니다.
- 미래의 실제 보상 신호를 $(\gamma\lambda)$의 비율로 빠르게 줄이는 대신 그 빈자리를 가치 함수 모델의 예측값($V$)으로 채우므로 가치 함수가 부정확할 때만 편향이 도입됩니다.

##### $0 < \lambda < 1$ (Compromise)
- $\lambda$를 중간값으로 설정하면 분산을 대폭 낮추면서도 가치 함수가 합리적으로 정확할 경우 편향을 허용 가능한 수준으로 유지할 수 있습니다.

### 3. GRPO (Group Relative Policy Optimization)
#### PPO의 문제점
#### 1. 보상의 희소성 (Sparse Reward)
LLM이 문장을 생성할 때, 보상 모델(Reward Model)은 보통 전체 답변이 완성된 후 마지막 토큰에만 점수를 부여합니다. 하지만 가치 모델은 특정 상태($s_t$, 즉 현재까지 생성된 토큰들)에서 앞으로 얻을 기대 보상을 예측해야 합니다. 하지만 실제 데이터에는 중간 단계의 보상이 없고 오직 마지막에만 결과가 나타나기 때문에, 각 중간 토큰이 최종 보상에 얼마나 기여했는지 직접적으로 알 수 있는 정답(Ground Truth)이 없습니다.  
따라서 정답 데이터가 마지막에만 집중되어 있으므로, 중간 단계의 모든 토큰에 대해 정확한 가치($V$)를 예측하도록 가치 모델을 훈련시키는 것은 매우 복잡하고 어려운 작업이 됩니다.

### 2. 신용 할당 문제 (Credit Assignment Problem)
행동(특정 토큰 선택)과 그 결과(최종 보상) 사이의 시간적 간격이 매우 길 때 발생하는 문제를 '신용 할당 문제'라고 합니다. 수백 개의 토큰으로 구성된 답변에서 최종적으로 높은 점수를 받았을 때, 과연 어떤 토큰이 결정적인 역할을 했는지 판별하기 어렵습니다.

### 3. 가치 모델의 부정확성이 미치는 영향 (편향 발생)
가치 모델이 모든 토큰에 대해 정확한 가치를 예측하지 못하면 다음과 같은 치명적인 문제가 발생합니다. PPO 알고리즘에서 가치 모델은 어드밴티지($A_t$)를 계산하는 기준점이 됩니다. 만약 가치 모델이 부정확한 값을 내놓으면 정책 업데이트 방향에 편향이 생기게 됩니다. 분산(Variance)은 더 많은 샘플로 해결할 수 있지만, 부정확한 가치 모델로 인한 편향은 알고리즘을 잘못된 방향으로 수렴하게 하거나 아예 수렴하지 못하게 만들 수 있습니다.

#### 4. 메모리 및 연산 자원 낭비
가치 모델은 보통 정책 모델과 크기가 비슷합니다. 만약에 이를 제거함으로써 학습에 필요한 메모리 사용량을 획기적으로 줄일 수 있습니다.

#### GRPO의 알고리즘
이러한 PPO의 까다로운 점들 때문에 새로운 알고리즘 GRPO는 가치 모델을 아예 제거하는 방식으로 DeepSeek 연구진이 수학적 추론 능력을 높이기 위해 제안한 PPO의 변형 알고리즘입니다.

GRPO(Group Relative Policy Optimization)는 PPO의 변형으로 PPO에서 어드밴티지를 계산하기 위해 필수적이었던 가치 모델(Critic)과 이를 이용한 GAE(Generalized Advantage Estimation) 방식의 어드밴티지 추정을 사용하지 않는 대신 동일한 질문에 대해 생성된 여러 샘플링 출력의 평균 보상을 베이스라인(baseline)으로 사용합니다.

구체적으로, 각 질문 $q$에 대해 GRPO는 이전 정책 $\pi_{\theta_{\text{old}}}$로부터 출력 그룹 $\{o_1, o_2, \cdots, o_G\}$를 샘플링한 후, 다음 목적 함수를 최대화하여 정책 모델을 최적화합니다.

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O \mid q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\lvert o_i \rvert} \sum_{t=1}^{\lvert o_i \rvert} \lbrace \min \lbrack \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})} \hat{A}_{i,t}, \;\text{clip}(\frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})}, 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \rbrack \rbrace - \beta \mathbb{D}_{\text{KL}} \lbrack \pi_{\theta} \| \pi_{\text{ref}} \rbrack \right]
$$

아래와 같이 GRPO는 PPO와 달리 수정된 KL-Divergence 항을 사용합니다. 이 KL 발산 항은 학습 중인 정책이 레퍼런스 정책(보통, 초기 SFT 모델)으로부터 너무 멀어지지 않도록 강제합니다. 이 KL 발산 항은 손실함수의 어드밴티지 항 뒤에 위치해서 만약 새로운 정책이 참조 정책과 너무 달라지면 KL 값이 커지게 되고, 이는 전체 목적 함수 값을 낮추는 결과(페널티)를 초래합니다. 따라서 모델은 보상을 최대화하는 동시에 참조 정책과의 유사성을 유지하는 방향으로 학습됩니다.

$$
\mathbb{D}_{\text{KL}} [\pi_\theta \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,\lt t})}{\pi_\theta(o_{i,t}|q, o_{i,\lt t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,\lt t})}{\pi_\theta(o_{i,t}|q, o_{i,\lt t})} - 1
$$

이 KL 발산항은 오리지널 KL 발산의 무편향 추정량(Unbiased estimator)으로 기댓값을 구했을 때 우리가 원래 구하려던 KL 발산과 일치하게 되는 식이 됩니다.  
$r = \frac{\pi_{\text{ref}}(o)}{\pi_\theta(o)}$라고 할 때, $r - \log r - 1$의 형태가 되며 이 항의 평균을 구해서 각 항을 분리해서 보면 아래와 같습니다.

**첫 번째 항**

$$
\mathbb{E}_{\pi_\theta} \left[ \frac{\pi_{\text{ref}}(o)}{\pi_\theta(o)} \right] = \sum_{o} \pi_\theta(o) \cdot \frac{\pi_{\text{ref}}(o)}{\pi_\theta(o)} = \sum_{o} \pi_{\text{ref}}(o) = 1
$$

**두 번째 항**

$$
\mathbb{E}_{\pi_\theta} \left[ \log \frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} \right] = D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

원래의 KL 발산 항입니다.

**세 번째 항**  
1

이를 통해 첫번째, 두번째, 세번째 항을 합해서 $1 - \log r - 1 = \log r$ 로 KL 발산항이 됩니다.

어드밴티지 항인 $\hat{A}_{i,t}$는 어떻게 계산할까요? 각 질문 $q$에 대해, 이전 정책 모델 $\pi_{\theta_{\text{old}}}$로부터 출력 그룹 $\{o_1, o_2, \cdots, o_G\}$를 샘플링합니다. 그런 다음 보상 모델을 사용하여 출력들에 대한 점수를 매기고, 이에 상응하는 $G$개의 보상 $r = \{r_1, r_2, \cdots, r_G \}$를 얻습니다. 그 후, 출력 내 모든 토큰의 어드밴티지 $\hat{A}_{i,t}$를 다음과 같이 정규화된 보상으로 설정합니다.

$$
\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}
$$

PPO 와 같이 이 목표 함수를 어드밴티지를 이용해 최대화함으로 정책을 최적화 합니다.

**계산 과정 정리**
1. 현재 정책 모델로부터 G개의 출력 $\{o_1, o_2, \cdots, o_G\}$을 샘플링합니다.
2. 보상 모델($r$) 을 통해 각 출력에 대한 보상 $r=\{r_1, r_2, \cdots, r_G\}$ 을 얻습니다.
3. 이 보상들을 그룹 내에서 정규화(Normalization)하여 최종 어드밴티지($\hat{A}$)를 구합니다.

**Reward Model($r$) of GRPO (vs $r$ of PPO)**  
PPO와 GRPO 모두 보상 모델(Reward Model)은 학습된 신경망(Neural Reward Model)을 사용하여 생성된 답변의 품질을 수치화된 점수로 변환하는 역할을 합니다. GRPO의 경우 DeepSeekMath-Base 7B를 기반으로 학습된 신경망으로, 답변의 최종 결과나 중간 추론 단계에 대해 수치화된 점수를 부여하여 그룹 내 상대적 어드밴티지를 계산할 수 있도록 합니다.  
다만 PPO는 그 점수를 해석하기 위해 '가치 모델'이라는 또 다른 AI를 옆에 두는 방식이고, GRPO는 여러 답변의 점수를 서로 비교하는 통계적 방식을 택해 자원을 절약한다는 점이 다릅니다.

### 4. GDPO (Group reward-Decoupled normalization Policy Optimization) 알고리즘
GDPO는 GRPO를 다중 보상(Multi-reward) 환경에 적용할 때 발생하는 '보상 신호 붕괴(reward collapse)' 문제를 해결하기 위해 NVIDIA 연구진에서 제시한 방법론으로 여러 보상 항목(예: 정확도, 형식, 길이 등)을 단순히 합산한 뒤 정규화하는 GRPO와 달리, 각 보상 항목을 개별적으로 정규화한 후 합산 한 후 전체 배치 단위로 한 번 더 정규화하여 안정성을 높이는 것이 특징입니다.

### Multi Rewards (다중 보상 in GRPO & GDPO)
GRPO는 구조적으로 $n$개의 보상 모델로부터 오는 신호를 수용할 수 있도록 설계되었고 최근 트렌드에서 그렇게 활용되고 있습니다. 이를 통해 GRPO는 여러 목표와 목표에 부합하는 보상을 처리할 수 있습니다.

$$
r_{\text{sum}}^{(i,j)} = r_1^{(i,j)} + \dots + r_n^{(i,j)}
$$

위 GRPO 보상 함수 수식은 $n$ 개의 목표(objectives)가 있을 때, 한 질문($i$)에 대한 $j$ 번째 응답에 대한 집계된 보상을 의미합니다. 가령 GDPO 논문처럼 도구 호출(형식 + 정확도), 수학 추론(정확도 + 길이 제약), 코딩 추론(통과율 + 길이 + 버그 비율) 등의 보상 모델들을 설정할 수 있습니다.

### Problem of GRPO: reward signal collapse in multi-reward RL
다중 보상 환경에서 GRPO 방식은 아래 식처럼 모든 개별 보상을 먼저 합산(Sum)한 뒤, 그 합계 점수를 그룹 내에서 정규화합니다.

$$
r_{\text{sum}}^{(i,j)} = r_1^{(i,j)} + \dots + r_n^{(i,j)}
$$

$$
A_{\text{sum}}^{(i,j)} = \frac{r_{\text{sum}}^{(i,j)} - \text{mean}\lbrace r_{\text{sum}}^{(i,1)}, \ldots, r_{\text{sum}}^{(i,G)} \rbrace}{\text{std}\lbrace r_{\text{sum}}^{(i,1)}, \ldots, r_{\text{sum}}^{(i,G)} \rbrace}
$$

이 경우 서로 다른 보상 성분들이(정확도, 형식, 길이 등등) 하나로 뭉쳐서 정규화하면, 서로 다른 품질의 답변들이 동일한 어드밴티지(Advantage) 값으로 변환되는 현상이 발생합니다. 그로 인해 모델은 여러 다른 품질의 답변들 중 어느 답변이 더 우수한지에 대한 정교한 차이를 구분하지 못하게 됩니다. 이를 "보상 신호 붕괴(reward signal collapse)" 라고 합니다.

#### Example
> **Gist 사용 시:** 아래 이미지도 절대 URL로 교체해 주세요.
>
> ![GDPO reward_signal_collapse](https://via.placeholder.com/500x300?text=GDPO+reward+signal+collapse)

예를 들어, 위 그림 처럼 (0점, 1점) 조합과 (0점, 2점) 조합을 GRPO와 GDPO에서 비교해 보겠습니다.

**GRPO**  
(0점, 1점)

$$
r_{\text{sum}}^{(1)} = r_{\text{obj1}}^{(1)} + r_{\text{obj2}}^{(1)} = 0 + 0 = 0
$$

$$
r_{\text{sum}}^{(2)} = r_{\text{obj1}}^{(2)} + r_{\text{obj2}}^{(2)} = 1 + 0 = 1
$$

$$
\mu = \frac{0 + 1}{2} = 0.5,\quad s^2 = \frac{(0 - 0.5)^2 + (1 - 0.5)^2}{2 - 1} = 0.25 + 0.25 = 0.5,\quad s = \sqrt{0.5} \approx 0.7071
$$

$$
A_{\text{sum}}^{(1)} = \frac{0 - 0.5}{0.7071} \approx -0.7071,\quad A_{\text{sum}}^{(2)} = \frac{1 - 0.5}{0.7071} \approx 0.7071
$$

$$
(-0.7071,\; 0.7071)
$$

(0점, 2점)

$$
r_{\text{sum}}^{(1)} = r_{\text{obj1}}^{(1)} + r_{\text{obj2}}^{(1)} = 0 + 0 = 0
$$

$$
r_{\text{sum}}^{(2)} = r_{\text{obj1}}^{(2)} + r_{\text{obj2}}^{(2)} = 1 + 1 = 2
$$

$$
\mu = \frac{0 + 2}{2} = 1.0,\quad s^2 = \frac{(0 - 1.0)^2 + (2 - 1.0)^2}{2 - 1} = \frac{1 + 1}{1} = 2.0,\quad s = \sqrt{2} \approx 1.4142
$$

$$
A_{\text{sum}}^{(1)} = \frac{0 - 1.0}{1.4142} \approx -0.7071,\quad A_{\text{sum}}^{(2)} = \frac{2 - 1.0}{1.4142} \approx 0.7071
$$

$$
(-0.7071,\; 0.7071)
$$

위 예시에서와 같이 GRPO에서 (0, 1) 조합의 정규화된 어드밴티지는 (−0.7071, 0.7071)이 됩니다. 마찬가지로 (0, 2) 조합 역시 정규화 과정을 거치면 똑같이 (−0.7071, 0.7071)이라는 값을 갖게 됩니다. 보상의 절대적인 차이(0과 2의 차이)가 더 커졌음에도 불구하고, 그 차이가 커진 만큼 표준편차도 함께 커지기 때문에 최종적으로 정규화된 어드밴티지 값은 동일하게 유지되는 것입니다.

이로 인해 GRPO에서는 "2점짜리 답변"과 "1점짜리 답변" 을 동일한 강도로 강화함으로써 학습 효율을 떨어뜨리는 "보상 신호 붕괴" 현상을 발생시키게 됩니다.

반면 GDPO의 경우...

**GDPO**  
(0점, 1점)

$$
r_{\text{obj1}}^{(1)} = 0,\quad r_{\text{obj1}}^{(2)} = 1
$$

$$
\mu_1 = \frac{0 + 1}{2} = 0.5,\quad s_1^2 = \frac{(0 - 0.5)^2 + (1 - 0.5)^2}{2 - 1} = 0.25 + 0.25 = 0.5,\quad s_1 = \sqrt{0.5} \approx 0.7071
$$

$$
A_1^{(1)} = \frac{0 - 0.5}{0.7071} = -0.7071,\quad A_1^{(2)} = \frac{1 - 0.5}{0.7071} = 0.7071
$$

$$
r_{\text{obj2}}^{(1)} = 0,\quad r_{\text{obj2}}^{(2)} = 0
$$

$$
A_2^{(1)} = 0,\quad A_2^{(2)} = 0
$$

$$
A_{\text{sum}}^{(1)} = -0.7071,\quad A_{\text{sum}}^{(2)} = 0.7071
$$

$$
(-0.7071,\; 0.7071)
$$

(0점, 2점)

$$
r_{\text{obj1}}^{(1)} = 0,\quad r_{\text{obj1}}^{(2)} = 1
$$

$$
\mu_1 = \frac{0 + 1}{2} = 0.5,\quad s_1 = \sqrt{0.5} \approx 0.7071
$$

$$
A_1^{(1)} = -0.7071,\quad A_1^{(2)} = 0.7071
$$

$$
r_{\text{obj2}}^{(1)} = 0,\quad r_{\text{obj2}}^{(2)} = 1
$$

$$
\mu_2 = \frac{0 + 1}{2} = 0.5,\quad s_2 = \sqrt{0.5} \approx 0.7071
$$

$$
A_2^{(1)} = -0.7071,\quad A_2^{(2)} = 0.7071
$$

$$
A_{\text{sum}}^{(1)} = -1.4142,\quad A_{\text{sum}}^{(2)} = 1.4142
$$

$$
(-1.4142,\; 1.4142)
$$

두 예시를 통해 GRPO와 GDPO를 비교해 보았을 때, GRPO는 다중 보상을 처리할 때 신호를 압축해버려 모델이 "무엇을 더 잘했는지" 구분하기 어렵게 만든 반면, GDPO는 보상 간의 구분을 유지함으로써 더 정확한 정책 업데이트(Accurate policy updates)와 우수한 수렴 성능을 이끌어냅니다.

### GDPO Methods
GRPO의 한계(다중 보상 처리시 보상 신호 붕괴)를 극복하기 위해 제안된 GDPO 알고리즘을 설명합니다. 기존 GRPO는 모든 보상을 먼저 더한 뒤 그룹 정규화를 수행하여 정보 손실(붕괴)을 야기했습니다. 반면, GDPO는 각 보상 항목을 개별적으로 정규화한 뒤 합산하는 방식을 취합니다. 이는 답변 간의 미세한 차이(fine-grained differences)를 보존하여 더 정확한 학습 신호를 제공하기 위함입니다.

$$
A^{(i,j)}_n = \frac{r^{(i,j)}_n - \text{mean}\lbrace r^{(i,1)}_n, \ldots, r^{(i,G)}_n \rbrace}{\text{std}\lbrace r^{(i,1)}_n, \ldots, r^{(i,G)}_n \rbrace}
$$

$$
A_{\text{sum}}^{(i,j)} = A_1^{(i,j)} + \dots + A_n^{(i,j)}
$$

위 식에서 보시다 시피 먼저 개별 보상 항목별 그룹 정규화를 한 후 통합합니다. 즉, 질문 $i$ 에 대한 $j$ 번째 응답의 $k$ 번째 보상 항목($r_k$) 에 대해서만 그룹 내 평균을 빼고 표준편차로 나눈 이후 이 정규화된 어드밴티지들을 하나로 통합합니다.

$$
\hat{A}_{\text{sum}}^{(i,j)} = \frac{A_{\text{sum}}^{(i,j)} - \text{mean}\lbrace A_{\text{sum}}^{(i',j')} \mid i' \in D_{\text{Batch}}, j' = 1, \dots, G \rbrace}{\text{std}\lbrace A_{\text{sum}}^{(i',j')} \mid i' \in D_{\text{Batch}}, j' = 1, \dots, G \rbrace + \epsilon}
$$

그 이후 합산된 어드밴티지를 전체 배치($D_{\text{Batch}}$)에 대해 다시 한번 정규화합니다. 이를 통해 최종 어드밴티지의 수치 범위를 일정하게 유지하여 학습의 안정성을 높입니다. 실제로 이 단계를 생략하면 가끔 학습이 수렴하지 못하고 실패하는 경우가 발생합니다.

#### Effective incorporation of priority variation: Conditioned Rewards
여러 보상 항목 사이에 Priority 가 다를 때, 이를 어떻게 효과적으로 모델 학습에 반영할 것인가에 대한 전략을 다룹니다. 일반적으로 여러 목표가 있을 때 각 보상에 가중치($w$)를 부여하여 중요도를 조절합니다.

$$
A^{(i,j)}_{\text{sum}} = w_1 A_1^{(i,j)} + \cdots + w_n A_n^{(i,j)}
$$

이 방법은 보상 항목 간의 난이도 차이(Difficulty disparity)가 클 경우, 가중치 조절만으로는 한계가 있습니다. 예를 들어 '답변 길이 조절'이 '수학 정답 맞히기'보다 훨씬 쉽다면, 모델은 가중치가 낮더라도 더 얻기 쉬운 '길이 보상'을 먼저 극대화하려 하며 정작 중요한 '정확도'는 무시하는 현상이 발생합니다.

$$
r_k = \lbrace \quad r_k \;\text{ if } r_l \geq t, \qquad 0 \;\text{ otherwise.} \quad \rbrace
$$

난이도가 낮은 보상이 학습을 지배하는 문제를 해결하기 위해, Conditioned Rewards 라는 더 강력한 설계 방식을 제안합니다. 위 수식 처럼 쉬운 보상($r_k$) 을 어려운 보상($r_l$)에 종속시킵니다. 즉, 어려운 목표를 달성했을 때만 쉬운 목표에 대한 보상을 주는 방식입니다. 예를들어, "답이 맞았을때만($r_{\text{correct}} = 1$), 길이가 짧은 것에 대한 보상($r_{\text{length}} = 1$)을 준다." 는 식입니다.  
이 방식을 쓰면 모델은 쉬운 보상을 챙기기 위해서라도 반드시 어려운 보상(인간이 우선순위를 둔 목표)을 먼저 달성해야만 합니다. 이는 모델이 쉬운 보상만 챙기는 Reward hacking을 원천적으로 방지합니다.

기존의 조건부 보상 없이 단순히 가중치만 줄였을 때는 모델의 행동이 예측 불가능하거나, 가중치를 아주 극단적으로 낮추어야만 겨우 효과가 나타났습니다. 하지만 Conditioned Rewards 을 도입한 후에는 가중치를 조금만 조절해도 모델의 성능(정확도 vs 효율성)이 매우 예측 가능하고 정교하게(Fine-grained) 변화하는 것을 확인했습니다.

### 5. PPO vs GRPO vs GDPO 비교
TBD

| 비교 항목 | PPO | GRPO | GDPO |
| :--- | :--- | :--- | :--- |
| **가치 모델(Critic)** | **필수** (별도 모델 필요) | **없음** (그룹 통계로 대체) | **없음** (그룹 통계로 대체) |
| **어드밴티지 방식** | 가치 함수 기반 GAE | 그룹 내 상대적 표준 점수 | 보상 항목별 개별 정규화 후 합계 |
| **자원 효율성** | 낮음 (메모리 부담 큼) | 매우 높음 (효율적 수렴) | 매우 높음 (다중 보상에 효율적) |
| **다중 보상 처리** | 합산 후 GAE 적용 가능 | 합산 후 정규화 (정보 손실 위험) | **개별 정규화 (높은 정밀도 보존)** |
| **주요 장점** | 가장 범용적이고 안정적 | 단일 목표(수학 등) 학습에 최적화 | 도구 호출 등 복합 목표 학습에 탁월 |

요약하자면, **PPO**는 안정적인 업데이트를 위한 표준 기법이며, **GRPO**는 가치 모델을 없애 효율성을 극대화한 DeepSeek의 혁신이고, **GDPO**는 GRPO의 효율성을 유지하면서 다중 보상 학습의 정밀도를 보완한 NVIDIA의 개선안입니다.
