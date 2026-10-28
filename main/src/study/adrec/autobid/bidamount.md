# BidAmount 이해하기
## 한 줄 요약

**BA(BidAmount)는 "광고주가 과금 액션 1번에 내는 돈"이면서, 동시에 "이 트래픽이 이 광고에 얼마나 가치 있나"를 나타내는 값** 입니다.
자동입찰에서는 이 BA를 시스템이 텀(10분)마다 다시 계산하고, 랭커는 그 값에 **상한(max_ba)** 을 씌워서 eCPM을 계산합니다.

| 용어 | 한 줄 뜻 | 단위 | 직관 |
|---|---|---|---|
| **base_bid** | 자동입찰 방문 목적의 BA | CPC | 클릭당 최대 금액 |
| **conv_base_bid** | 자동입찰 전환 목적의 BA | CPA | 전환당 최대 금액 |
| **max_ba** | 전환 목적에서 클릭당 BA의 상한 | CPC | pCVR 과예측 방어용 클릭당 최대 금액 |
| **base_ba** | 랭커가 광고 목적에 맞춰 고른 BA (위 둘 중 하나, 또는 광고주 입력값) | 목적에 따라 | |
| **pricing_adjusted_ba** | base_ba를 실제 과금 타입 단위로 바꾼 값 | 과금 타입 | 최대 과금액 |
| **ranker_ba** | 할인(SR)까지 반영해 최종적으로 광고주가 내는 금액 | 과금 타입 | 로그의 `bidAmount` |

이 문서는 아래 세 위키를 순서대로 정리하되, [eCPM 이해하기](../ecpm/ecpm.md)와 [자동입찰 이해하기](autobid.md)에 이미 있는 내용은 빼고 **새로 추가되는 내용만** 담았습니다.

1. [1.1. 자동입찰 입찰가 계산 기본](https://kakao.atlassian.net/wiki/spaces/adrec/pages/140808548)
2. [1.2. BidAmount](https://kakao.atlassian.net/wiki/spaces/adrec/pages/144478502)
3. [1.3. MaxBa](https://kakao.atlassian.net/wiki/spaces/adrec/pages/140824486)

---

## 1. 입찰가 계산 기본 (1.1)

### 1.1 pAR: BA에 짝지어지는 확률

eCPM 식은 [eCPM 이해하기 4.1](../ecpm/ecpm.md)과 같습니다. 이 위키에서는 BA와 곱하는 확률을 **pAR**(predicted Action Rate)이라고 부릅니다.

$$
\text{eCPM} = \text{bidAmount} \times \text{pAR} \times 1000
$$

| 과금 타입 | BA의 뜻 | pAR |
|---|---|---|
| CPM | 노출 1,000회당 광고주가 DSP에 내는 돈 | pVR |
| CPC | 클릭당 광고주가 DSP에 내는 돈 | pCTR |
| CPA | 전환당 광고주가 DSP에 내는 돈 | pCVR |

> ecpm.md의 es, 이 위키의 pAR, 로그의 `pcr`은 모두 같은 자리("BA에 곱하는 확률")를 가리킵니다. 위키 표는 CPA의 pAR을 pCVR로 적었지만, 노출 기준으로는 pCTR × pCVR이 곱해집니다(ecpm.md 4.2).

### 1.2 실제 로그로 eCPM 읽기

자동입찰 CPC 광고가 클릭된 로그에서 입찰가 부분만 뽑으면 이렇습니다.

```json
"winner": {
  "chargeAmount": 154.0,
  "bidAmount": 154,
  "pcr": 0.0016620439291000367,
  "ecpm": 255.9556884765625,
  "bidStrategy": "AUTOBID",
  "pricing": "CPC",
  "ranking": { "ad": "{\"pctr\":0.001662044, \"ba\":154, ...}" }
}
```

| 필드 | 값 | 읽는 법 |
|---|---|---|
| bidStrategy | AUTOBID | 자동입찰 |
| pricing | CPC | 클릭 과금 → pAR로 **pCTR**을 씀 |
| pcr | 0.00166 | 과금 타입에 맞춰 고른 확률. 여기선 `ranking.ad.pctr`과 같은 값 |
| bidAmount | 154 | 클릭당 154원 |
| ecpm | 255.96 | 154 × 0.00166 × 1000 = **255.96원** |
| chargeAmount | 154.0 | 클릭이 났기 때문에 과금됨. 클릭이 안 나면 **0.0**으로 남음 |

### 1.3 수동입찰과 자동입찰의 eCPM 비교

같은 요청, 같은 pCTR 0.02라고 해 봅시다.

| | BA를 정하는 방법 | eCPM |
|---|---|---|
| 수동입찰 | 광고주가 어드민에서 클릭당 **200원** 입력 | 200 × 0.02 × 1000 = **4,000원** |
| 자동입찰 | 시스템이 실시간으로 `f(budget / request)` 계산 → 지금은 **100원** | 100 × 0.02 × 1000 = **2,000원** |

자동입찰의 eCPM 식은 결국 이렇게 쓸 수 있습니다.

$$
\text{eCPM} = \underbrace{f\!\left(\frac{\text{budget}}{\text{request}}\right)}_{\text{bidAmount (실시간 계산)}} \times \text{pAR} \times 1000
$$

### 1.4 경매 방식에 따른 광고주 과금액

[자동입찰 이해하기 3장](autobid.md)에서 FPA·SPA를 DSP가 내는 돈 관점에서 봤다면, 이번엔 **광고주가 내는 돈(chargeAmount)** 관점입니다.

| 경매 | 광고주 과금액 | 뜻 |
|---|---|---|
| FPA | `ChargeAmount = BidAmount` | 할인 없이 BA 그대로 과금 |
| SPA | `ChargeAmount = BidAmount × second_eCPM / eCPM` | 2등과의 eCPM 비율만큼 BA를 깎아서 과금 |

**예시 (SPA)**

내 광고: BA 300원(CPC), eCPM 3,000원. 2등의 eCPM은 2,400원.

$$
\text{ChargeAmount} = 300 \times \frac{2{,}400}{3{,}000} = 240\text{원}
$$

클릭이 나면 300원이 아니라 **240원**이 과금됩니다. "2등만 이길 만큼만 낸다"는 SPA 원리를 클릭당 가격으로 바꾼 것입니다.

### 1.5 자동입찰 BA의 입력값

자동입찰 BA는 네 가지 값의 함수입니다.

$$
\text{bidAmount} = f(\text{budget},\ \text{spend},\ \text{request},\ \text{bid})
$$

| 입력 | 뜻 | 어디서 오나 | BA에 주는 영향 |
|---|---|---|---|
| budget | 일예산. 하루 최대 사용 금액 | 광고주가 모먼트 어드민에서 입력 (요일·시간대별 집행 시간도 설정) | 많을수록 **BA ↑** |
| spend | 지금까지 쓴 돈 | 입찰 결과 | (남은 예산 계산에 사용) |
| request | 단위 시간(10분, 1시간 등)당 광고 요청 수 | 광고주가 고른 타겟(성별·연령·지면 등)에 해당하는 유저의 요청 | 많을수록 기회가 많아 **BA ↓** 가능 |
| bid | 단위 시간당 실제로 입찰에 응한 수 | 입찰 결과 | |

"예산은 많은데 요청은 적다" → 요청 하나하나를 더 비싸게 사도 되므로 BA를 올립니다. "예산은 적은데 요청은 많다" → 골라 사면 되므로 BA를 내립니다.

---

## 2. BidAmount 용어와 계산 방법론 (1.2)

### 2.1 BA는 두 가지 뜻으로 쓰인다

- 과금 타입에 따라 **광고주가 내는 금액**
- 트래픽에 대해 **광고별로 매기는 가치(value)**

그래서 광고추천 안에서는 BA가 단계마다 이름을 바꿔 가며 등장합니다.

### 2.2 단계별 BA 용어

| 용어 | 정의 | 직관 |
|---|---|---|
| **base_bid** | 자동입찰 **방문** 목적 eCPM 계산에 쓰는 CPC BA. PDM 할인 때문에 최종 과금액보다 **크거나 같음** | 클릭당 최대 금액 (방문 목적에선 max_ba와 같은 뜻) |
| **conv_base_bid** | 자동입찰 **전환** 목적 eCPM 계산에 쓰는 CPA BA | 전환당 최대 금액. 클릭당 상한은 max_ba가 따로 맡음 |
| **base_ba** | 랭커가 광고 목적(goal)에 따라 고른 BA | 아래 표 |
| **pricing_adjusted_ba** | base_ba를 **실제 과금 타입(bidType) 단위**로 바꾼 값. PDM의 SR 환원으로 줄어들 수 있음 | 최대 과금액 |
| **ranker_ba** (rankerBidAmount) | 랭커 → 이데아로 넘기는 **최종 광고주 과금액**. PDM의 SR까지 반영 | CPM이면 노출 1,000건당, CPC면 클릭당, CPA면 전환당 과금액 |

base_ba는 이렇게 정해집니다.

| bidStrategy | goal | base_ba | 단위 |
|---|---|---|---|
| MANUAL | (무관) | 광고주 입력값 | bidType에 맞춤 |
| AUTOBID | VISITING | base_bid | CPC |
| AUTOBID | CONVERSION | conv_base_bid | CPA |

pricing_adjusted_ba는 base_ba의 단위와 bidType이 같으면 base_ba 그대로이고, 다르면 bidType 단위로 바꿉니다. 대표적인 경우가 **자동입찰 전환 목적**입니다. CPA 단위 BA를 클릭 과금(CPC)으로 바꿔야 합니다.

$$
\text{pricing\_adjusted\_ba (CPC)} = \min\big(\text{conv\_base\_bid (CPA)} \times \text{pCVR},\ \text{max\_ba (CPC)}\big)
$$

> 위키 원문은 `base_ba(CPA) * pctr`로 적혀 있지만, CPA를 CPC로 바꾸려면 클릭→전환 확률인 pCVR을 곱해야 하고 1.3 MaxBa 위키의 랭커 코드도 `conv_base_bid * pcvr`입니다. 여기서는 pCVR로 정리했습니다.

**예시: 자동입찰 전환 목적, 클릭 과금**

| 단계 | 계산 | 값 |
|---|---|---|
| base_ba = conv_base_bid | 자동입찰 계산값 | 전환당 20,000원 |
| CPC로 환산 | 20,000 × pCVR 2% | 클릭당 400원 |
| pricing_adjusted_ba | min(400, max_ba 300) | 클릭당 **300원** |
| eCPM | 300 × pCTR 1% × 1000 | 3,000원 |
| ranker_ba | 300원에서 PDM의 SR 할인 반영 | 300원 이하 |

- 로그의 `rawMessage.winner.bidAmount`와 `rawMessage.winner.rankerBidAmount`는 거의 같게(±1) 남습니다.
- 그래서 **흔히 외부에서 말하는 "BA"는 로그의 `winner.bidAmount` = ranker_ba**라고 생각하면 됩니다.

### 2.3 Ba Calculator: 자동입찰 BA는 어디서 계산되나

- 수동입찰은 광고주 입력값으로 BA가 **고정**됩니다.
- 자동입찰은 **Autopilot**이 BA를 **가변**으로 계산합니다. 2.2의 base_bid, conv_base_bid가 그 결과입니다.

```
[Autopilot] 텀(10분)마다 배치로 모든 자동입찰 광고그룹의 BA 계산
     ↓ autobid_meta에 저장
[랭커] AUTOBID 전체 광고그룹 메타파일을 주기적으로(약 1분?) 읽음
     ↓ 트래픽이 오면
     광고별 BA = autobid_meta 값 → eCPM 계산
```

랭커와 별개로 자동입찰이 노리는 것은 하나입니다. **텀마다 BA를 갱신해서 예산만큼을 정확히 소진하는 BA를 찾는 것.** (위키에도 "주기가 정말 1분이면 더 줄여 달라고 요청 필요"라는 메모가 있습니다.)

| 용어 | 뜻 |
|---|---|
| **텀** (term) | 예산과 BA가 갱신되는 시간 단위 (현재 10분) |
| **피드백** (feedback) | 이전 BA로 입찰한 결과들. spend, e_spend, bid_rate, imp, click 등을 일정 기간 집계해서 봄 |
| **state** | 한 순간에 BA 계산에 쓰는 피처 전체. 피드백 외에 온갖 정보가 들어갈 수 있음 |

BA 계산 방식은 크게 **피드백 컨트롤러 방식**(결과를 보고 고치기)과 **예측기 방식**(상태를 보고 바로 맞히기)으로 나뉩니다.

### 2.4 BA 계산 방법론: 표현 방식 세 가지

여기부터는 Autopilot 구현이 아니라, **일반적으로 BA를 어떻게 계산·갱신할 수 있는지**에 대한 정리입니다. 대부분의 방법은 아래 세 표현 중 하나(또는 조합)입니다.

| 표현 | 식 | 하는 일 | 예시 |
|---|---|---|---|
| **절대값 계산형** | `ba' = f(state)` | 상태를 보고 BA를 직접 산출 | `target_CPA × k`, 선형 모델, GBDT |
| **곱셈 업데이트형** | `ba' = ba × update_weight(state)` | 기존 BA를 "조금 올리거나 내리는" 컨트롤러 | `update_weight = clamp(target_CPA / actual_CPA, 0.5, 1.5)` |
| **Score → BA 매핑형** | `score = S(state)`, `ba = g(score)` | 내부 가치·위험도·공격성 같은 score를 만든 뒤 매핑 함수로 BA 변환 | 선형 `λ × score`, ORTB 계열 `C × (sqrt(1 + λ·score²) − 1)` |

[자동입찰 이해하기 6.5](autobid.md)의 `BA = f(B/T)`는 절대값 계산형이고, PID 보정은 곱셈 업데이트형에 해당합니다.

### 2.5 얼마나 자주 적용하나

같은 방법이라도 적용 단위에 따라 의미가 달라집니다.

| 단위 | 하는 일 | 예 |
|---|---|---|
| 실시간 (트래픽마다) | 요청·경매마다 BA 계산 | `ba = f(pCTR, pCVR, 유저·슬롯 피처, 텀 상태 …)` |
| 텀 / 단기 | 일정 기간마다 광고 단위 배율 조정 | 텀별 spend·CPA 오차를 보고 다음 텀 BA 배율 조정 |
| 배치 (하루~며칠) | 광고별 고정 BA를 길게 갱신 | 지난 7일 성과로 다음 주 BA 계산 |

### 2.6 방법론 계열

| 계열 | 아이디어 | 장점 | 단점·주의 |
|---|---|---|---|
| **룰 기반** | KPI가 목표에서 벗어난 비율·구간을 보고 BA를 곱셈 조정 | 구현 쉽고 직관적, 디버깅 쉬움. 초기에 가장 많이 씀 | 세밀한 최적화는 어려움 |
| **제어 이론** (P/PI/PID, LQG/MPC) | "목표 − 실제" 오차로 피드백 제어 | 과소·과대 집행과 BA 변동성을 균형 있게 다룸 | LQG/MPC는 튜닝이 어려움 |
| **응답곡선 / Bid Landscape** | 과거 로그로 BA ↔ spend·click·CPA 관계 곡선을 그리고 거기서 최적 BA를 찾음 | 광고그룹별 특성 반영, 해석 쉬움 | 데이터 희소성, 시계열 변화에 민감 |
| **지도학습** | state → BA를 회귀·분류 모델로 직접 학습 | 배치 갱신에 적합 | 좋은 라벨(정답 BA)을 만들기 어려움 |
| **Bandit / Bayesian Optimization** | BA 후보를 arm(또는 연속 변수)으로 두고 탐색·활용 | 신규·데이터 부족 광고의 안전한 탐색 | 갱신 빈도가 낮을 때 유리 |
| **강화학습** | 시간에 따른 BA **시퀀스 전체**를 정책으로 학습 | 장기 최적화 | 서비스 적용 시 안전장치·시뮬레이션 필수 |

각 계열을 대표 식 위주로 보면 이렇습니다.

**룰 기반**

```python
# KPI 비율 기반 스케일링
cpa_ratio     = actual_CPA / target_CPA
update_weight = clamp(1 / cpa_ratio, 0.5, 1.5)
ba_new        = ba_old * update_weight

# 구간(band) 기반: 벗어나면 강하게, 안이면 유지
if actual_CPA > 1.2 * target_CPA:   ba_new = ba_old * 0.8
elif actual_CPA < 0.8 * target_CPA: ba_new = ba_old * 1.2
else:                               ba_new = ba_old

# 스케줄/패턴 반영: 요일·시간 프로파일 위에 작은 보정
ba_new = time_profile[day_of_week, hour] * update_weight_from_KPI
```

예: 목표 CPA 10,000원인데 실제 CPA가 12,500원이면 `cpa_ratio = 1.25`, `update_weight = 0.8` → BA를 20% 내립니다.

**제어 이론**

- **P / PI 제어**

```python
error_t = target_metric - actual_metric
I_t     = I_{t-1} + error_t                 # 적분: 오차 누적
control = Kp * error_t + Ki * I_t
update_weight = exp(control)  # 또는 1 + control
ba_new  = ba * clamp(update_weight, min_w, max_w)
```

  error 합, 기울기, 이동평균 등 state를 풍부하게 쓸수록 더 안정적입니다.

- **LQG / MPC**: 예산·성과·시간을 상태공간 모델(상태 $x_t$ = [누적 소진, 남은 예산, 오차 합, …], 제어 $u_t$ = BA)로 두고, 일정 기간(horizon) 동안 아래 비용을 최소화합니다.

$$
J = \sum_t (\text{spend}_t - \text{target}_t)^2 + \rho\,(\Delta \text{ba}_t)^2
$$

  앞 항은 "목표만큼 쓰기", 뒤 항은 "BA를 너무 자주·크게 바꾸지 않기"입니다. ρ로 둘의 균형을 정합니다.

**응답곡선 / Bid Landscape**

- **경험적 곡선 피팅**: 지난 N일 BA 수준별 spend·click·CPA를 모아 `spend(ba)`, `CPA(ba)`를 근사한 뒤 푼다.
  - 예산 기준: $\min_{ba} |\text{spend}(ba) - \text{target\_budget}|$
  - 성과 기준: $\max_{ba} \text{clicks}(ba)$ s.t. $\text{spend}(ba) \le \text{budget}$
- **Dual / Lagrange**: 여러 광고그룹 $g$의 전체 문제를 라그랑지안으로 쓰고, λ 하나를 조정하면서 그룹별 BA를 정합니다. 전체 예산을 관리해야 하는 시스템에 자연스러운 구조입니다.

$$
\max_{ba} \sum_g \text{clicks}_g(ba_g) - \lambda \left( \sum_g \text{spend}_g(ba_g) - \text{Budget} \right)
$$

  λ는 [자동입찰 이해하기 5.2](autobid.md)의 "예산 1원의 그림자 가격"과 같은 역할입니다.

**지도학습**

- **회귀**: 광고 메타, 지난 k일 성과(spend, click, CPA, ROAS, 변동성), 요일·시간 패턴 → BA. 라벨은 응답곡선 최적화로 얻은 "pseudo-optimal BA"나, 사람이 튜닝해서 성과가 좋았던 기간의 BA.
- **구간 + 분류**: BA를 {100, 200, 300, …} 같은 bin으로 나누고 "좋았던 bin"을 라벨로 학습. 안정적이고 최소·최대 BA 제약을 걸기 쉬움.
- **기본식 + 보정치**: 사람이 만든 식에 ML 보정을 얹음.

```python
ba_base = target_CPA * k
delta   = model(state)        # log 스케일 보정
ba_new  = ba_base * exp(delta)
```

**Bandit / Bayesian Optimization**

- **비컨텍스트 Bandit**: 한 광고그룹에 BA 후보 여러 개(arm)를 두고, 텀·배치마다 하나를 골라 reward(클릭, 매출, ROAS − 예산 초과 페널티 등)를 보고 갱신. ε-greedy, UCB, Thompson Sampling.
- **Contextual Bandit**: 광고그룹 상태(성과, 메타, 시즌)를 context로 써서 상황별로 좋은 arm을 학습.
- **Bayesian Optimization**: BA를 연속값으로 보고 `f(ba) = 성과`를 블랙박스 함수로 두고 탐색. 갱신이 드문 배치·대형 캠페인에 적합.

**강화학습**

| | Batch-level RL | 실시간 RL |
|---|---|---|
| 한 step | 배치 하나(예: 하루)의 BA 결정 | 요청·텀 단위 BA 결정 |
| state | 최근 n일 성과, 남은 예산·기간, 시즌 | 텀 상태 등 |
| action | BA 또는 BA 배율 | BA |
| reward | 기간 동안의 클릭·매출 − 예산·변동성 페널티 | |
| 알고리즘 | DQN(이산 BA), PPO/A2C(연속 BA), Offline RL | |
| 주의 | | 최소·최대 BA 같은 안전장치, 지도학습 warm-start, 오프라인 시뮬레이션·리플레이가 필수 |

### 2.7 실제 서비스는 하이브리드

실서비스에서는 위 방법들을 층으로 쌓아 씁니다.

```
┌─ Safety / Guard ─────────── ba_min·ba_max, 최소·최대 CPA·ROAS 제약,
│                              급격한 변화 제한, blacklist/whitelist
├─ Main Policy ────────────── 룰 / PI·MPC / 응답곡선 / 지도학습 / Bandit·BO / RL 중 하나 또는 조합
├─ Smoothing & Stabilization ─ ba' = α·candidate_ba + (1−α)·ba, 변화율 제한, 이동평균, momentum
└─ Exploration (옵션) ──────── 작은 범위 랜덤화, bandit 기반 탐색
```

3장의 max_ba가 바로 이 구조의 **Safety / Guard 레이어**에 해당합니다.

### 2.8 최신 논문 흐름 (2022–2025)

흐름은 **강화학습 → 시퀀스 모델링(Decision Transformer) → 생성 모델(Diffusion) → 대규모 사전학습 모델**입니다. 공통 목표는 RL 대비 성능 향상과, 오프라인 로그만으로 안전하게 학습하는 실용성입니다.

| 연도 | 논문 (학회) | 계열 | 핵심 아이디어 | 보고된 성과 |
|---|---|---|---|---|
| 2023 | **MBAB** (AAMAS) | 모델 기반 RL | 경매를 확률 모델로 거칠게 근사하고 동적계획법으로 핵심 조정 변수를 온라인 튜닝 | 시뮬레이터 없이 실서비스(Alibaba/Ant)에 적용 |
| 2024 | **DiffBid** / AIGB (KDD) | Diffusion | 목표 리턴(RTG)을 조건으로 **입찰 시퀀스 전체를 한 번에 생성** | Alibaba A/B: GMV +2.81%, ROI +3.36% |
| 2024 | **GAS** (arXiv) | 생성 정책 + 탐색 | 사전학습한 DT 정책 출력에 MCTS 스타일 후처리 탐색, 선호도별 평가자 투표 | Kuaishou: 목표 CPA 대비 비용 사용 +4.60% |
| 2025 | **EBaReT** (WWW) | Decision Transformer | 전문가 궤적 생성 + PU 판별기로 로그의 전문가 행동만 따라 하기, 구간(bag) 단위 보상 재분배 | Kuaishou: 전환 +13.46%, 매출 +11.44% |
| 2025 | **Optimal RTG-DT** (WWW) | Decision Transformer | DT의 RTG를 사람이 정하지 않고 상태별 최적 RTG를 예측, 고보상 궤적으로 데이터 확대 | 공개 시뮬레이션에서 기존 DT보다 일관되게 우수 |
| 2025 | **GAVE** (SIGIR) | 생성형 오프라인 RL | KPI 제약을 점수로 RTG에 반영, value function으로 검증한 탐색 행동만 반영 | NeurIPS 2024 AIGB 대회 1위 |
| 2025 | **GRAD** (preprint) | 대규모 사전학습 | Causal Transformer + MoE로 후보 행동 다양화, 가치 예측기로 제약 고려 | Meituan: 거래액 +2.18%, ROI +10.68% |
| 2025 | **BERT-BidRL** (preprint) | LLM + RL | 사전학습 Transformer를 상태 인코더로 쓰고 PPO로 학습, CPA 제약은 보상 페널티 + 제약 인지 디코더 | DT·DiffBid 대비 전환수↑, CPA 편차↓ |
| 2025 | **CBD** (arXiv) | Diffusion | Completer가 실제로 가능한 미래 상태를 생성, Aligner가 목표에 맞게 궤적을 보정 | 희소 보상 환경에서 DiffBid 대비 전환가치 +29.9% |

- 많은 논문이 공개 데이터셋 **AuctionNet**(Alibaba, 약 5억 건 경매 로그)으로 비교합니다.
- 성과 수치는 각 논문의 자체 보고값이며, 위키 원문의 요약을 옮긴 것입니다.

---

## 3. MaxBa: BA의 상한 (1.3)

### 3.1 랭커는 항상 max_ba를 씌운다

랭커(Danke Ranker)의 eCPM 계산 코드(24.03.25 기준)를 식으로 옮기면 이렇습니다.

$$
\text{eCPM} = \min(\text{BA} \times \text{weight},\ \text{max\_ba}) \times \text{es} \times 1000
$$

- **weight**: COS 정보, 전환 목적 정보, BA 정보로 정하는 랭킹 가중치
- **es**: CPC이면서 자동입찰이면 `pctr_bid_power`를 거듭제곱으로 적용. keyword search 지면은 `modelWeight`를 더함(그 외 지면은 0이라 영향 없음)

pCVR 사용 여부와 관계없이 **항상 max_ba로 상한을 겁니다.** 경우별로 풀면 이렇습니다.

| 경우 | eCPM |
|---|---|
| 방문, 또는 전환인데 valid group off (pCVR 미사용) | `min(base_bid, max_ba) × pCTR × 1000` |
| 전환이고 valid group on (pCVR 사용) | `min(conv_base_bid × pCVR, max_ba) × pCTR × 1000` |

valid group은 [eCPM 이해하기 4.4](../ecpm/ecpm.md)의 전환 목적 CPC에서 "pCVR을 믿고 쓸 만한 광고그룹"을 뜻합니다.

그럼 max_ba는 경우마다 어떻게 정할까요?

```
goal == VISITING ─────────────────────────────▶ 3.2  max_ba = base_bid
goal == CONVERSION
  ├─ valid group off (pCVR 없음/timeout, 직전 텀 win 0) ─▶ 3.3
  └─ valid group on (pCVR 있음)
       ├─ 비용목표 (tCPA, tROAS) ─────────────────▶ 3.4  cpc_max_ba
       └─ 전환수 최대화 ──────────────────────────▶ 3.5  cpc_max_ba와 pid_max_ba의 가중평균
```

### 3.2 방문 목적: max_ba = base_bid

방문 목적에서는 `max_ba ≥ base_bid`이기만 하면 상한이 아무 일도 안 합니다. 그래서 간단히 **max_ba = base_bid**로 둡니다.

> 위키 메모: 현재 코드는 방문도 3.3의 valid group off 로직으로 계산 중이라, 문제가 없는지 확인이 필요하다고 적혀 있습니다.

### 3.3 전환 목적, valid group off

goal은 CONVERSION인데 pCVR을 못 쓰는 경우입니다(pCVR 예측기가 안 돌거나, 값이 timeout으로 안 들어오거나, 직전 텀 win이 0).

**방문처럼 max_ba = base_bid로 두지 않는 이유**

지금은 pCVR이 없어서 eCPM에 max_ba가 영향을 주지 않지만, **텀 중간에 valid group이 on으로 바뀌면** 갑자기 pCVR이 곱해지면서 BA가 튈 수 있습니다. 그때 과소진을 막도록 max_ba를 **미리 적당한 값으로 준비**해 두는 것입니다.

```python
# valid group off, 또는 직전 텀 win == 0
if prev_4_term_cpc_ba > 0:
    ba_avg = prev_4_term_cpc_ba     # 최근 4텀 실제 클릭당 BA
elif n_day_cpc > 0:
    ba_avg = n_day_cpc              # 최근 n일 CPC
else:
    ba_avg = base_bid

max_ba = max(ba_avg * non_pcvr_ba_tolerance, base_bid)   # 최소 base_bid는 보장
max_ba = min(max_ba, BASE_BID_HARD_MAX)                  # 절대 상한
```

기본 생각은 "텀 소진율이 1.0일 때 `max_ba = cpc_ba × ba_tolerance`"이고, cpc_ba가 없으면 n_day_cpc, 그것도 없으면 base_bid로 대신합니다.

**예시**: 최근 4텀 클릭당 BA 250원, `non_pcvr_ba_tolerance` 1.5, base_bid 300원이면 `max(250 × 1.5, 300) = max(375, 300)` → **max_ba = 375원** (HARD_MAX보다 작다고 가정).

### 3.4 전환 목적, valid group on: 비용목표 (tCPA, tROAS)

비용목표 상품은 텀 소진율을 보지 않으므로 CPC 기반 상한만 씁니다.

$$
\text{max\_ba} = \text{cpc\_ba} \times \text{ba\_tolerance}
$$

여기서 ba_tolerance는 고정값이 아니라 **전환 수에 따라 바뀝니다**.

**왜 tolerance를 전환 수에 따라 바꾸나**

- valid group이 off → on으로 막 바뀐 광고는 예측기가 학습한 전환이 **최소 1건**뿐일 수 있어 pCVR을 과예측하기 쉽습니다. → 상한을 **좁게**.
- 전환이 많이 쌓인 광고는 pCVR이 더 정확하다고 보고 → 상한을 **넓게**.

**동적 ba_tolerance**: 최근 7일 전환 수(`prev_7d_conv`)를 x로, 두 임계값 사이를 선형 보간합니다.

```
ba_tolerance
  ▲
 y3 ┤                        ●━━━━━━━━  max_ba_tolerance
    │                  ╱
 y2 ┤          ●──╱               static_ba_tolerance
    │        ╱
 y1 ┤──╱                          min_ba_tolerance
    └──────────┴─────────────┴──────▶ 전환 수 (prev_7d_conv)
              x1             x2
     static_conv_threshold   max_conv_threshold
```

| 구간 | ba_tolerance |
|---|---|
| x < x1 | y1에서 y2로 선형 증가 |
| x1 ≤ x < x2 | $y_2 + (y_3 - y_2) \cdot \dfrac{x - x_1}{x_2 - x_1}$ |
| x ≥ x2 | y3 (고정) |

- 실제 config는 y3를 직접 받지 않고, `static_ba_tolerance`(y2)와 `max_ba_tolerance_adjustment`로 y3를 계산합니다.
- 위키 코드의 첫 구간은 `y1 + x/(x1 - x)`로 적혀 있지만, 그래프는 y1 → y2 직선입니다. 그래프대로라면 $y_1 + (y_2 - y_1) \cdot x / x_1$이므로, 실제 값은 코드로 확인이 필요합니다.

**예시** (숫자는 설명용): y1 = 1.0, y2 = 1.5, y3 = 2.0, x1 = 10건, x2 = 50건.

| 최근 7일 전환 | ba_tolerance | cpc_ba 200원일 때 max_ba |
|---|---|---|
| 5건 | 1.0 + 0.5 × 5/10 = 1.25 | 250원 |
| 30건 | 1.5 + 0.5 × 20/40 = 1.75 | 350원 |
| 80건 | 2.0 | 400원 |

### 3.5 전환 목적, valid group on: 전환수 최대화

전환수 최대화는 **지난 텀 소진율**까지 보고 max_ba를 정합니다. 후보가 두 개입니다.

| 후보 | 식 | 뜻 |
|---|---|---|
| **cpc_max_ba** | `cpc_ba × ba_tolerance` | 소진율 1.0일 때의 기준 상한 (3.4와 같음) |
| **pid_max_ba** | `prev_max_ba × pid_output` | 소진율을 반영한 PID 출력으로 지난 상한을 늘리거나 줄인 값 |

pid_output은 소진 상황을 반영한 배율로, 아래 그래프상 미소진이면 커지고(max_ba ↑) 과소진이면 작아집니다(max_ba ↓).

**왜 pid_max_ba가 필요한가**

- **미소진일 때**: conv_base_bid는 오르는데 max_ba가 안 따라 오르면, 상한에 막혀 계속 미소진 → conv_base_bid가 더 빨리 오르는 악순환이 생깁니다.
- cpc_max_ba는 랜덤 노출이나 낮은 cpc_ba로 이긴 경우들 때문에 평균이 잘 안 오르는 문제도 있습니다.
- **과소진일 때**도 max_ba가 너무 낮아지지 않도록, 1.0보다 작은 쪽에도 적용합니다.

**가중평균**: pid_output이 1 근처(pt2 ~ pt3)면 cpc_max_ba를 그대로 쓰고, 많이 벗어날수록(pt1 이하, pt4 이상) pid_max_ba로 넘어갑니다. 그 사이는 섞습니다.

```
weight (cpc_max_ba 쪽 비중)
 1.0 ┤        ┌────┐
     │       ╱      ╲
 0.0 ┤──────┘        └────────
     └──────┴──┴────┴──┴──────▶ pid_output
           pt1 pt2  pt3 pt4
```

```python
pid_max_ba = prev_max_ba * pid_output
cpc_max_ba = cpc_ba * ba_tolerance

weight = min((pid_output - pt1) / (pt2 - pt1), (pt4 - pid_output) / (pt4 - pt3))
weight = min(max(weight, 0), 1)
max_ba = pid_max_ba * (1 - weight) + cpc_max_ba * weight

# 변화량 제한: 한 텀에 prev_max_ba 대비 ±limit 이상 못 움직임
lower_bound = (1 - max_ba_change_rate_limit) * prev_max_ba
upper_bound = (1 + max_ba_change_rate_limit) * prev_max_ba
max_ba = min(max(max_ba, lower_bound), upper_bound)
```

pid_max_ba 자체가 pid_output에 비례하므로, 최종 max_ba를 pid_output에 대해 그리면 섞이는 구간(pt1~pt2, pt3~pt4)이 직선이 아닌 곡선이 됩니다.

**예시** (숫자는 설명용): prev_max_ba 320원, cpc_max_ba 350원, pt = (0.8, 0.95, 1.05, 1.2), 변화량 제한 20%(256 ~ 384원).

| pid_output | 상황 | weight | 계산 | 최종 max_ba |
|---|---|---|---|---|
| 1.0 | 소진 정상 | 1 | cpc_max_ba | **350원** |
| 1.1 | 약간 미소진 | (1.2 − 1.1) / 0.15 ≈ 0.67 | 352 × 0.33 + 350 × 0.67 | **약 351원** |
| 1.3 | 크게 미소진 | 0 | pid_max_ba = 320 × 1.3 = 416 → 상한 384에 걸림 | **384원** |
| 0.7 | 과소진 | 0 | pid_max_ba = 320 × 0.7 = 224 → 하한 256에 걸림 | **256원** |

변화량 제한이 있어서 한 텀에 max_ba가 크게 튀지 않습니다. 2.7의 **Smoothing & Stabilization 레이어**와 같은 역할입니다.

---

## 4. 용어 사전

[eCPM 이해하기](../ecpm/ecpm.md)와 [자동입찰 이해하기](autobid.md)의 용어 사전에 없는 것만 모았습니다.

| 용어 | 뜻 |
|---|---|
| pAR | predicted Action Rate. BA에 곱하는 확률 (CPM: pVR, CPC: pCTR, CPA: pCVR). ecpm.md의 es, 로그의 pcr |
| chargeAmount | 실제 광고주 과금액. 과금 이벤트가 안 나면 0.0 |
| second_eCPM | 경매 2등의 eCPM. SPA 과금액 계산에 사용 |
| base_bid | 자동입찰 방문 목적 BA (CPC) |
| conv_base_bid | 자동입찰 전환 목적 BA (CPA) |
| base_ba | 랭커가 목적에 따라 고른 BA (광고주 입력값 / base_bid / conv_base_bid) |
| pricing_adjusted_ba | base_ba를 과금 타입 단위로 바꾼 값. 최대 과금액 |
| ranker_ba (rankerBidAmount) | SR까지 반영한 최종 광고주 과금액. 로그의 `winner.bidAmount`와 거의 같음 |
| Autopilot | 자동입찰 BA를 텀마다 배치로 계산하는 시스템 |
| autobid_meta | Autopilot 계산 결과. 랭커가 주기적으로 읽음 |
| 피드백 | 이전 BA로 입찰한 결과 (spend, e_spend, bid_rate, imp, click …) |
| state | BA 계산에 쓰는 피처 전체 |
| ORTB | Optimal RTB. score를 `C·(sqrt(1 + λ·score²) − 1)` 꼴로 BA에 매핑하는 계열 |
| LQG / MPC | 상태공간 모델로 목표 오차와 BA 변화량 비용을 함께 최소화하는 제어 |
| Bid Landscape | BA와 spend·click·CPA의 관계 곡선 |
| Decision Transformer (DT) | RL 문제를 시퀀스 생성 문제로 푸는 Transformer |
| RTG | Return-To-Go. 앞으로 얻고 싶은 누적 보상. DT·Diffusion 입찰 모델의 조건 |
| AuctionNet | Alibaba가 공개한 대규모 경매 로그 데이터셋 |
| max_ba | 클릭당 BA 상한. eCPM 계산 시 항상 `min(BA, max_ba)` |
| valid group | pCVR을 서빙에 써도 될 만큼 전환이 쌓인 광고그룹 |
| ba_tolerance | 평균 cpc_ba의 몇 배까지 상한을 허용할지 정하는 배율. 전환 수에 따라 동적으로 정함 |
| cpc_ba | 실제 클릭당 BA (예: 최근 4텀 평균) |
| cpc_max_ba | `cpc_ba × ba_tolerance`. 소진율 1.0 기준 상한 |
| pid_output | 텀 소진율을 반영한 PID 출력 배율 |
| pid_max_ba | `prev_max_ba × pid_output` |
| pt1 ~ pt4 | pid_output 임계값. cpc_max_ba와 pid_max_ba의 섞는 비율을 정함 |
| max_ba_change_rate_limit | 한 텀에 max_ba가 움직일 수 있는 최대 비율 |
| BASE_BID_HARD_MAX | max_ba의 절대 상한 |
