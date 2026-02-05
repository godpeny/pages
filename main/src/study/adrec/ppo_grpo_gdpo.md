# 0. Preliminary
<b> $\pi_{\theta(a_t|s_t)}$ - 정책 함수</b>  
상태 $s_t$가 주어졌을 때 행동 $a_t$ 를 선택할 조건부 확률 분포 -> 모델이 현재 상황($s_t$)에서 어떤 행동($a_t$​)을 취할 가능성이 얼마나 높은지를 수치화합니다.  

<b> $A^\pi(s_t, a_t)$ - 어드밴티지 함수 </b>  
정책 $\pi$ 에 따른 어드밴티지 함수로 특정 행동 $a_t$가 정책의 평균적인 행동보다 얼마나 더 좋은지를 나타냅니다.
$$
A^\pi(s_t, a_t) := Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

- $Q^\pi(s_t, a_t)$: 상태 $s_t$ 에서 특정 행동 $a_t$ 를 취한 다음, 그 이후로 부터는 정책 $\pi$를 따랐을 때의 총 보상의 합. = State-Value Function
- $V^\pi(s_t)$: 상태 $s_t$ 에서 단순히 현재 정책 $\pi$ 를 따랐을 때 기대되는 총 보상의 합. = Action-Value Function
- $A^\pi(s_t, a_t)$: 내가 지금 한 행동($a_t$) 이 평소 하던 대로($\pi$) 했을 때보다 얼마나 더 좋은가?"를 측정하는 것.

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
g^\gamma := \mathbb{E}_{\substack{s_{0:\infty} \\ a_{0:\infty}}} \left[ \sum_{t=0}^\infty A^{\pi,\gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t) \right], \quad A^{\pi,\gamma}(\mathbf{s}_t, \mathbf{a}_t) := Q^{\pi,\gamma}(\mathbf{s}_t, \mathbf{a}_t) - V^{\pi,\gamma}(\mathbf{s}_t).
$$

강화학습에서는 보상을 높여야 하므로 위에서 예측한 기울기(Gradient) 방향으로 매개변수를 더해주는 Gradient Ascent 방법을 사용합니다. 
$$\theta_{i+1} \leftarrow \theta_i + \alpha_i g_i$$
- $\theta_i$: 현재 단계($i$)의 정책함수의 파라미터입니다.
- $\theta_{i+1}$: 업데이트된 다음 단계의 파라미터입니다.
- $\alpha_i$: 학습률(Learning rate)로, 한 번의 업데이트에서 정책을 얼마나 변경할지 결정합니다.
- $g_i$**: 파라미터 공간에서 추정된 기울기로 높은 보상과 연결된 행동의 발생 확률을 높이는 방향을 가리킵니다.

# 2. PPO (Proximal Policy Optimization)
## PPO 알고리즘
PPO도 Policy Gradient Method 의 한 종류로 Gradient Ascent을 통해 미분 가능한 정책 함수($\pi$)에 의해 action이 결정되는 강화학습(RL) 에이전트를 훈련하는 데 사용됩니다. 일반적인 Policy Gradient Method는 정책 업데이트($\theta_{i+1} \leftarrow \theta_i + \alpha_i g_i$) 단계를 거쳐 에이전트가 기대 보상을 점점 더 높이는 방향으로 학습하지만 이런 일반적인 Policy Gradient method는 불안정할 수 있습니다. 즉, 보폭(step size = learning rate)이 너무 크면 정책이 최적이지 않은 방향으로 미끄러지며(drift) 최적화로의 복구가 거의 불가능해집니다. 그렇지만 반대로 보폭이 너무 작으면 전반적인 훈련 효율성이 떨어집니다.

이러한 불안정성을 해결하기 위해, PPO는 에이전트의 정책 업데이트가 한 번의 step에 너무 커지지 않도록 제한하는 "clip 함수"를 도입합니다. 이를 통해 Gradient Ascent 과정에서 미끄러짐(drift)에 대한 위험 없이 더 큰 보폭을 사용할 수 있습니다. PPO의 손실 함수는 다음과 같이 정의됩니다.

$$
\mathcal{J}_{\text{PPO}}(\theta)
= \mathbb{E}\!\left[
\frac{1}{|o|} \sum_{t=1}^{|o|}
\min\!\Big(
\frac{\pi_\theta(o_t \mid q, o_{<t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})} A_t,\,
\text{clip}\big(\frac{\pi_\theta(o_t \mid q, o_{<t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})}, 1 - \varepsilon, 1 + \varepsilon\big) A_t
\Big)
\right]
$$

- $\pi_\theta, \pi_{\theta_{\text{old}}}$: 신규 및 이전 정책 모델입니다,.
- $\pi_\theta(o_t \mid q, o_{<t})$: 입력 프롬프트 $q$와 이전에 생성된 모든 토큰 $o_{<t}$ 가 주어졌을 때, 토큰 $o_t$를 생성할 확률입니다.
- $q, o$: 각각 질문 데이터셋과 이전 정책 $\pi_{\theta_{\text{old}}}$에서 샘플링된 질문과 출력입니다.
- $\varepsilon$: 훈련 안정화를 위해 PPO에서 도입된 클리핑 관련 하이퍼파라미터입니다.
- $A_t$: GAE(Generalized Advantage Estimation)를 적용하여 계산된 어드밴티지(advantage)입니다.
- **clip**
  - $\frac{\pi_\theta(o_t \mid q, o_{<t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})}> 1 + \varepsilon$: 새 모델이 동일한 행동에 대해 너무 높은 확률을 부여할 때 → 이를 클리핑(제한)합니다. 즉, 만약 어떤 행동이 보상이 좋아서(Advantage > 0) 신규 정책($\pi_\theta(o_t \mid q, o_{<t})$) 이 이 행동을 할 확률을 계속 높이다가, 그 비율이 $1 + \varepsilon$ 을 넘어서게 되면 제한(clip)이 작동합니다.
  - $\frac{\pi_\theta(o_t \mid q, o_{<t})} {\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})} < 1 - \varepsilon$: 너무 낮은 확률을 부여할 때 → 이 역시 클리핑합니다. 반대로 보상이 나빠서 확률을 낮추다가 비율이 $1 - \varepsilon$ 밑으로 떨어져도 마찬가지로 제한이 걸립니다.
  - Clip 함수가 기존 정책 대비 신규 정책의 확률 비율을 제한하는 이유는 모델의 파라미터($\theta$)가 너무 멀리 이동하는 것을 막기 위해서입니다.

PPO는 이처럼 clip 방식을 사용하여 새 정책을 이전 정책과 **"가깝게(close)"** 유지함으로써 파괴적인 대규모 업데이트를 방지합니다. 이것이 기존의 기본적인 정책 경사법과 비교하여 PPO의 훈련을 안정화하는 핵심 요소입니다,.

KL-패널티(KL-penalty)가 포함된 보상 모델은 다음과 같이 정의됩니다.

$$
r_t = r_\phi(q, o_{\le t}) - \beta \log \frac{\pi_\theta(o_t \mid q, o_{<t})} {\pi_{\text{ref}}(o_t \mid q, o_{<t})}
$$

- $r_\phi$: 주어진 입력 $q$에 대해 모델의 출력 $o$가 얼마나 인간이 선호하는 것인지를 알려주는 스칼라 점수 $r_\phi$를 제공하는 학습된 보상 모델입니다,. 이는 별도로 학습되므로 PPO 과정 중에는 고정(fixed)됩니다.
- $\pi_{\text{ref}}$: 학습의 기준이 되는 고정된 모델을 의미하며, 일반적으로 강화학습 이전 단계인 지도 미세 조정(SFT)을 마친 초기 모델입니다.

즉, 첫 번째 항인 $r_\phi(q, o_{\le t})$는 더 나은 답변을 생성하도록 장려하는 반면, 두 번째 항(KL-패널티 = KL-발산)은 원래 모델에서 급격하게 변화하는 것을 억제합니다. 이를 통해 학습 중인 모델이 레퍼런스 모델(원래 모델)에서 너무 멀리 벗어나지 않도록(drift) 억제하여 학습의 안정성을 높이고 보상 모델의 과적합을 방지합니다.

이제 어드밴티지는 GAE(Generalized Advantage Estimation)를 통해 일련의 $r_t$ 값으로부터 계산됩니다,.

$$
A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k
\big( r_{t+k} + \gamma V_\psi(o_{t+k+1}) - V_\psi(o_{t+k}) \big), \quad 
V_\psi(o_t) \approx \mathbb{E}[R_t \mid o_t]
$$

여기서 $R_t$는 토큰 $o_t$에서 시작하는 기대 미래 누적 보상입니다. $V_\psi(o_t)$는 가치 모델(value model)로 단순히 현재 정책 $\pi$ 를 따랐을 때 기대되는 총 보상의 합을 의미합니다. 이는 현재 상황이 이미 얼마나 좋은지에 대한 베이스라인(baseline) 추정치를 제공합니다,.

결론적으로, <b> 보상(reward) → 어드밴티지(advantage) → PPO 최적화 </b>의 흐름으로 진행됩니다. 이해를 돕기 위해 관련 이미지를 함께 확인하시기 바랍니다.

## GAE에서 어드밴티지 함수 추정
실제 할인된 ($\gamma$ 가 적용된) 어드밴티지 함수 $A^\pi(s_t, a_t)$ 는 알 수 없으므로, 전체 궤적 데이터를 사용하여 계산한 근사치를 대신 사용해야하므로 이를 추정하는 방법을 다룹니다. 즉, ground truth 할인된 어드밴티지 함수($A^\pi(s_t, a_t)$)를 정확하게 추정하는 $\hat{A}$ 를 만드는 것입니다.

$$
\hat{g} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{\infty} \hat{A}_t^n \nabla_\theta \log \pi_\theta (a_t^n | s_t^n)
$$
실제 어드밴티지 $A$를 모르니 위와 같이 어드밴티지 추정치 $\hat{A}$ 를 만들어 배치 단위(N)로 평균을 내는 방식을 사용합니다.

추정치 $\hat{A}$ 를 만들 때는 TD-Residual 개념을 사용합니다. 근사 가치 함수 $V$를 사용하여 다음과 같이 정의 할 수 있습니다.

$$
\delta_t^{V^{\pi,\gamma}} = r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)
$$
위 식은 현재 상태 $s_t$ 에서 행동 $a_t$ 를 취했을 때 얻은 즉각적인 보상($r_t$) 과 다음 상태의 가치($\gamma V^{\pi,\gamma}(s_{t+1})$)의 합에서, 원래 예상했던 현재 상태의 가치($V^{\pi,\gamma}(s_t)$)를 뺀 값으로 나타납니다. 즉 TD-Reisudal은 행동 $a_t$ 에 대한 어드밴티지의 추정치로 간주할 수 있습니다.

이를 $k$ 타입스텝으로 확장하면,
$$
\hat{A}_t^{(1)} := \delta_t^V = -V(s_t) + r_t + \gamma V(s_{t+1}) \\[5pt]
\hat{A}_t^{(2)} := \delta_t^V + \gamma\delta_{t+1}^V = -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) \\[5pt]
\hat{A}_t^{(3)} := \delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V = -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}) \\[5pt]
\cdots \\[5pt]
\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta^{V}_{t+l} = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k}) \\[5pt]

\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{k-1} \gamma^l r_{t+l}
$$

위를 바탕으로 일반화된 어드밴티지 추정치(Generalized Advantage Estimator, GAE)를 정의합니다. GAE는 현재 시점($1$)에서 $\infin$ 단계까지의 어드밴티지 추정치들을 적절한 비율($\lambda$)로 지수적으로 가중 평균한 것입니다.
$$
\begin{align*}
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} &:= (1 - \lambda) (\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \ldots) \\
&= (1 - \lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \ldots) \\
&= (1 - \lambda)\left(\delta_t^V (1 + \lambda + \lambda^2 + \ldots ) + \gamma\delta_{t+1}^V (\lambda + \lambda^2 + \lambda^3 + \ldots ) \right. \\
& \quad \left. {} + \gamma^2\delta_{t+2}^V (\lambda^2 + \lambda^3 + \lambda^4 + \ldots ) + \ldots \right) \\
&= (1 - \lambda) \left( \delta_t^V \left( \frac{1}{1 - \lambda} \right) + \gamma\delta_{t+1}^V \left( \frac{\lambda}{1 - \lambda} \right) + \gamma^2\delta_{t+2}^V \left( \frac{\lambda^2}{1 - \lambda} \right) + \ldots \right) \\
&= \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V
\end{align*}
$$

즉 GAE는 TD residual을 이용하여 여러 개의 k-스텝 어드밴티지(Q-V) 추정치를 $\lambda$ 를 이용해 지수 가중 평균(weighted sum)하는 개념입니다.


### 3. GRPO (Group Relative Policy Optimization) 설명
DeepSeek 연구진이 수학적 추론 능력을 높이기 위해 제안한 PPO의 변형 알고리즘입니다.

*   **가치 모델(Critic) 생략**: 정책 모델과 크기가 비슷한 가치 모델을 별도로 두지 않아 **훈련 자원(메모리 및 계산량)을 획기적으로 절감**합니다.
*   **그룹 상대적 어드밴티지**: 각 질문($q$)에 대해 여러 개의 답변(Group $G$)을 샘플링한 뒤, 해당 그룹 내에서의 상대적 점수를 기반으로 어드밴티지를 계산합니다.
    *   $A_{i,j} = \frac{r_{i,j} - \text{mean}(r)}{\text{std}(r)}$
*   **학습 방식**: 답변의 최종 결과에 점수를 매기는 **결과 감독(Outcome Supervision)**이나 단계별로 점수를 주는 **과정 감독(Process Supervision)**을 통해 수학적 추론 효율을 극대화합니다.

### 4. GDPO (Group reward-Decoupled normalization Policy Optimization) 설명
NVIDIA에서 제안한 알고리즘으로, 여러 보상이 동시에 존재하는 **다중 보상(Multi-reward) 환경**에서의 GRPO 한계를 극복합니다.

*   **보상 신호 붕괴(Reward Collapse) 해결**: 기존 GRPO는 여러 보상을 단순히 합산한 뒤 정규화하므로, 서로 다른 보상 조합들이 동일한 어드밴티지 값으로 뭉쳐 정보가 손실되는 문제가 있었습니다.
*   **분리된 정규화(Decoupled Normalization)**: 각 보상 항목별로 그룹 정규화를 **먼저 수행**하여 개별 어드밴티지를 구한 뒤 이를 합산합니다. 이를 통해 보상들 사이의 미세한 차이(해상도)를 보존합니다.
*   **배치 정규화(Batch Normalization)**: 합산된 어드밴티지를 전체 배치 단위에서 다시 정규화하여 보상 항목의 개수와 상관없이 **수치적 안정성**을 유지합니다.

### 5. PPO vs GRPO vs GDPO 비교

| 비교 항목 | PPO | GRPO | GDPO |
| :--- | :--- | :--- | :--- |
| **가치 모델(Critic)** | **필수** (별도 모델 필요) | **없음** (그룹 통계로 대체) | **없음** (그룹 통계로 대체) |
| **어드밴티지 방식** | 가치 함수 기반 GAE | 그룹 내 상대적 표준 점수 | 보상 항목별 개별 정규화 후 합계 |
| **자원 효율성** | 낮음 (메모리 부담 큼) | 매우 높음 (효율적 수렴) | 매우 높음 (다중 보상에 효율적) |
| **다중 보상 처리** | 합산 후 GAE 적용 가능 | 합산 후 정규화 (정보 손실 위험) | **개별 정규화 (높은 정밀도 보존)** |
| **주요 장점** | 가장 범용적이고 안정적 | 단일 목표(수학 등) 학습에 최적화 | 도구 호출 등 복합 목표 학습에 탁월 |

요약하자면, **PPO**는 안정적인 업데이트를 위한 표준 기법이며, **GRPO**는 가치 모델을 없애 효율성을 극대화한 DeepSeek의 혁신이고, **GDPO**는 GRPO의 효율성을 유지하면서 다중 보상 학습의 정밀도를 보완한 NVIDIA의 개선안입니다.