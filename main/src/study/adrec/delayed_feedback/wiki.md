> **TL;DR**
> 1. 신규 광고그룹의 pCVR 안정화 시간 $N$ 을 먼저 측정한다.
> 2. Alibaba 논문 3종(ES-DFM, DEFER, DEFUSE) 도입 실험 + 옵티마이저 파라미터 튜닝을 하고 지표를 본다.
> 3. 그 외 GDFM, ULC, 신규 group_id 임베딩 초기화 개선을 추가로 실험해볼 수 있다.

**새로 만들거나 설정을 바꾼 광고그룹은 pCVR이 안정될 때까지 며칠 걸리고, 그 사이 과대·과소예측이 발생한다.** 그 시간을 측정해서 줄인다.

# 구조

| | 내용 |
|---|---|
| **목적** | 신규·설정변경 광고그룹의 캘리브레이션 도달 시간 $N$ 단축 |
| **제약 ①** | 성숙 구간 **캘리브레이션** 비열위 |
| **제약 ②** | 성숙 구간 **RIG** 비열위 |

"신규 그룹을 빨리 맞히게 만들되, 이미 잘 맞히던 그룹을 망가뜨리지 말 것." 무엇을 내주면 안 되는지가 미리 정해져 있어야 실험 결과를 판정할 수 있다. 가드레일을 AUC가 아니라 RIG로 두는 이유는 이 과제가 **캘리브레이션 문제**이기 때문이다 — AUC는 순위만 보므로 pCVR을 전부 2배로 올려도 변하지 않는다.

# $N$ 의 정의

> 그룹 생성(또는 설정 변경) 후 **캘리브레이션 오차가 ±X% 안에 Y일 연속 머무는 최초 시점**

- "연속 Y일"이 핵심 — 우연히 하루 맞은 것을 도달로 세지 않는다
- 누적 노출·클릭은 트래픽 규모가 다른 그룹을 비교하기 위한 보조 축

# 측정 방법

- **전체 트래픽 평균이 아니라 "그룹 생성 후 경과일"로 묶어서** 캘리브레이션 곡선을 그린다. 전체 평균은 성숙 그룹이 압도적으로 많아 신규 그룹의 어긋남이 씻겨 나간다
- 적응 속도는 **학습이 끝나는 속도가 아니라 예측이 맞아지기까지의 시간**( 캘리브레이션 오차가 ±X% 안에 Y일 연속 머무는 최초 시점)이다.
- 목적별로 따로 잰다 — PURCHASE는 배치 DFM, APP/PF/MEM은 온라인 FNW로 메커니즘이 다르다

# 아직 안 된 것

1. **$N$ 을 측정한 적이 없다** — 원인 후보를 논하기 전에 곡선부터 그려야 한다
2. **$N$ 일 동안의 손실액을 모른다** — 이 값 없이는 과제 우선순위를 정당화할 수 없다

# 예상 원인별 논문 근거
> 각 원인(C1~C4)에 대해 **문제점을 지적한 논문**과 **해결 방안을 제시한 논문**을 대표적인 것만 골라 정리한다. 인용은 원문 그대로 두고 한 줄 해석을 붙였다.

## C1 — 라벨 성숙

> **문제** : 전환은 클릭하자마자 오지 않는다. 며칠 뒤에 오기도 한다(최대 7일). 그래서 어제 클릭에 붙은 "전환 안 함"은 진짜 안 샀다는 뜻이 아니라 **아직 모른다**는 뜻이다. 배치 DFM은 이걸 알고 있어서 최근 클릭의 "안 샀다" 신호를 거의 안 믿는다 — 얼마나 믿을지가 $F(d) = 1 - e^{-\lambda d}$ 이고, $\lambda = 0.7/\text{일}$ 이면 2시간 전 클릭은 5.6%만 반영한다. 신규 그룹은 가진 데이터가 전부 "최근"이라 이 할인을 그대로 다 맞는다.  
온라인 모델(`fnw`)은 정반대로 일단 전부 "안 샀다"로 즉시 학습하기 때문에 첫 며칠은 무조건 과소예측이 난다. 전환이 실제로 며칠 걸려 오는 것 자체는 앞당길 수 없으니 **없앨 수 없는 최소 지연**이다. 다만 아래 논문들은 두 가지를 더 말한다 — (a) 초반에 보이는 라벨은 그냥 적기만 한 게 아니라 **빨리 사는 사람 쪽으로 기울어 있고**(C1-b), (b) 장바구니처럼 빨리 오는 신호를 쓰면 일부는 당겨올 수 있다.

### 문제점 지적

- **FSIW** (Yasui et al., WWW 2020) — https://arxiv.org/abs/2002.02068
  지연 피드백을 "학습 데이터와 서빙 환경의 조건부 라벨 분포 불일치"로 처음 정식화하고, 그 결과가 **항상 과소예측**임을 증명했다.
  > "some positive instances at the training period are labeled as negative because some conversions have not yet occurred when training data are gathered. As a result, the conditional label distributions differ between the training data and the production environment."
  > "a CVR predictor would be prone to downward bias under the feedback shift because … $P(Y=1 \mid X=x) \le P(C=1 \mid X=x)$."

- **Ktena et al.** (RecSys 2019) — https://arxiv.org/abs/1907.06558
  연속 학습에서 "기다리는 시간 vs fake negative 비율"이 트레이드오프임을 지적. 현행 온라인 DFOM(`fnw`)의 원 논문.
  > "fresh data may not have complete label information at the time they are ingested by the training algorithm. Naive strategies which consider any data point a negative example until a positive label becomes available tend to underestimate CTR"
  > "It is also unclear what the ideal window length would be, in order to find a trade-off between the delay in model training and the fake negative (FN) rate"

- **DLA-DF** (Saito, Morishita & Yasui, SIGIR 2020,) — https://arxiv.org/abs/1910.01847
  C1의 하위 문제: 초반에 관측되는 positive는 **빠른 전환자에 쏠려 있다**(MNAR). 라벨이 "적은" 게 아니라 "편향된" 것.
  > "decisive users are much more likely to convert immediately after a click than indecisive users. Therefore, the probabilities of conversions being observed correctly are not uniform among samples. … the MNAR mechanism can lead to sub-optimal and biased estimations"

- **DEFUSE** (Chen et al., WWW 2022) — https://arxiv.org/abs/2202.06472
  기존 재가중 방법들의 공통 결함: 관측 negative를 일괄 negative로 취급.
  > "observed negatives may potentially be fake negatives, and these methods falsely treat them as real negatives, leading to sub-optimal performance."

## C2 — 표본 희석 · 낡은 prior

> **문제** : 신규 그룹의 임베딩은 미학습 상태(콜드스타트)라 예측은 사실상 **과거 30일의 다른 그룹들로 학습된 공유 파라미터가 내놓는 "평균적인 그룹"의 값**이 되고, 그래서 이 그룹의 실제 CVR과 어긋난다. 둘로 갈린다 — **C2-①** 임베딩이 아직 안 배워짐(30일 창과 무관, 7일이어도 같음) / **C2-②** 그동안 받는 기본값(공유 파라미터)이 지난 30일 평균이라 **오늘과 어긋남**(창이 길수록 커짐). 학습 기간을 30→7일로 줄이면 비중은 오르지만 라벨 미성숙 샘플 비율도 올라 C1이 악화되므로, 위키 처방은 "기간을 자르지 말고 최근 데이터에 가중치".

### 문제점 지적

- **GDFM** (Yang & Zhan, NeurIPS 2022) — https://arxiv.org/abs/2206.00407
  C2를 **두 개로 쪼개 정식화**한 유일한 논문. sampling gap(표본 부족 = C2-①)과 temporal gap(낡은 분포 = C2-②).
  > "(i) Estimating conversion rates via post-click actions requires more samples than using conversion labels directly, which highlights the importance of sample complexity. (ii) The post-click actions bring information of past distributions, which incurs a temporal gap."
  > "Even if we have unlimited samples from $p_{t-\delta}$ we are only able to recover $p_{t-\delta}(y \mid x)$ instead of $p_t(y \mid x)$."
  → 표본이 무한해도 **과거 분포**밖에 못 배운다. 신규 그룹이 받는 "평균값"이 낡은 이유.

- **nnDF** (Kato & Yasui, KDD 2022,) — https://arxiv.org/abs/2009.13092
  모든 지연 보정이 깔고 있는 **정상성 가정**을 명시하고, **신규 캠페인이 그 가정을 깬다**고 직접 썼다.
  > "Assumption 2 (Stationarity Assumption). … $p(Y_i(E_i^t) \mid X_i = X, E_i^t = s) = p(Y_j(E_j^{t'}) \mid X_j = X, E_j^{t'} = s)$"
  > "This often happens in advertising platforms, such as when a new campaign is launched. Since there are fewer data of the new campaigns, the test data has some shift from the training data, so the stationarity assumption does not hold in this setting."
  → C1 보정이 C2 상황에서 틀린다. Criteo 실험에서 새 캠페인 비율에 따라 날마다 최고 방법이 바뀜.

- **ULC** (KDD 2023)
  재가중 계열은 **이미 있는 positive의 무게를 재분배**할 뿐이라, 과거와 다른 신규 그룹의 fake negative를 표현할 수 없다.
  > "[This problem] is worse when the data distribution has changed recently. As the information about the false negative samples may differ from the past observed positive samples, only using the observed positive samples cannot complement the correct information about the fresh false negative samples."

## C3 — 파라미터 갱신량

> **문제** : 데이터도 있고 라벨도 왔는데 **한 번 학습할 때 파라미터가 움직이는 폭이 작으면** 여전히 오래 걸린다. 온라인 학습률은 APP 1e-4, PF·MEM 1e-3이고 조정 스케줄이 없다. 배치는 ClippyAdagrad lr 0.01에 Adagrad 누적기가 run마다 리셋된다. **재학습 주기(배치 4시간·온라인 30분)는 원인이 아니다** — 자주 학습하는 것과 빨리 따라잡는 것은 다르고, 남는 변수는 회당 갱신량이다.

해당 논문 없음. 지연 피드백 문헌은 손실 함수와 라벨 설계를 다루고 학습률·옵티마이저는 다루지 않는다. **기존 파라미터와 lr 값을 확인한 뒤 미세조정으로 판정한다.**

## C4 — 초기 표현 (제안)

> **문제** : 새 그룹이 생기면 group_id 임베딩이 $N(0,\ 10^{-5})$ 에서 시작한다(`models/model/feature.py`의 `OneHotFeature`, `RN_STDDEV = 0.00001`). 그 그룹은 어느 계정·캠페인·목적인지 다 알려져 있는데 그 정보를 하나도 안 쓰고 **빈 종이로 출발**한다. C1~C3가 전부 완벽해도 **첫 예측은 출발점이 결정**하며, $N$ 은 그룹 생성 순간부터 재므로 첫 몇 시간은 이 문제다.

> **참고**: 콜드스타트 초기화를 다룬 논문은 MetaEmb(SIGIR 2019, CTR), **지연 피드백 문헌 밖**이다.

# 예상 원인별 해결 제안
> 각 원인에 대한 해결 방안을 모았다 — 각 방법이 **무엇**이고 논문이 무엇을 보였는지. 우리가 **무엇을 어떤 순서로** 하는지는 아래 결론.

## C1·C2 — 지연 피드백 보정

### ES-DFM — 대기 창을 설계 변수로

> **ES-DFM** (Yang et al., AAAI 2021) — https://arxiv.org/abs/2012.03245

**로직** — 클릭이 오면 정해둔 시간 $e$ 만큼 기다렸다가, 그 안에 전환이 왔으면 양성으로, 안 왔으면 음성으로 라벨을 붙여 흘려보낸다. 창이 닫힌 뒤에 전환이 도착하면 양성 복사본을 한 번 더 넣는다. 이렇게 만들어진 왜곡된 분포를 보조 모델 둘로 되돌린다.

**$p_{dp}$, $p_{rn}$ 정리**

기호
- $x$ — 클릭 하나의 피처(그룹·상품·유저·지면 등). pCVR 모델 입력 그대로
- $h$ — 클릭부터 전환까지 걸린 시간
- $e$ — 대기 창. 이만큼 기다렸다가 라벨을 붙인다. 설계 변수, 실무에서는 상수 하나(예: 1시간)

관측 시점에 클릭은 셋으로 나뉜다.

| | 확률 | 관측 라벨 | 처리 |
|---|---|---|---|
| 창 안 전환 | $p(y{=}1)\,p(h \le e)$ | 1 | 그대로 |
| 창 뒤 전환 (지연 양성) | $p(y{=}1)\,p(h > e) = p_{dp}$ | 0 → 나중에 1 복사본 | **보정 대상** |
| 진짜 미전환 | $p(y{=}0)$ | 0 | 그대로 |

정의

$$
p_{dp}(x) = p(y{=}1 \mid x)\,p(h > e \mid x, y{=}1)
$$

이 클릭이 결국 전환하는데, 그 전환이 **창 밖**에 오는 확률. $h > e$ 가 창 밖이다.

$$
p_{rn}(x) = \frac{p(y{=}0 \mid x)}{p(y{=}0 \mid x) + p_{dp}(x)}
$$

관측 시점에 0으로 보인 것(분모) 중 진짜 미전환(분자)의 비율.

어떻게 구하나 — 둘 다 **별도 이진 분류기** $f_{dp}(x)$, $f_{rn}(x)$ 로 학습한다. FNW처럼 CVR 모델 자기 출력을 갖다 쓰지 않는다. 학습 데이터는 라벨이 확정된 **과거 로그**(예: 30일 지난 스트림). 각 클릭이 창 안에 샀는지, 창 뒤에 샀는지, 끝까지 안 샀는지가 확정 사실이라 가짜 음성이 없다. 라벨은 아래처럼 붙인다.

| 클릭의 최종 결과 | $f_{dp}$ 라벨 | $f_{rn}$ 라벨 |
|---|---|---|
| 창 안 전환 | 0 | 제외 |
| 창 뒤 전환 | **1** | 0 |
| 끝까지 미전환 | 0 | **1** |

**분류기 $f_{dp}$, $f_{rn}$ 의 구조와 학습** — $p$ 는 진짜 확률(이론 유도용, 알 수 없음), $f$ 는 그것을 예측하도록 학습한 분류기의 출력(실제 가중치에 들어가는 값). pCVR의 $p(y{=}1 \mid x)$ 와 $f_\theta(x)$ 관계와 같다.

가중치는 양성에 $1 + p_{dp}$, 음성에 $(1 + p_{dp})\,p_{rn}$. 둘 다 확률이라 $[0,1]$ 안에서만 만들어진다. 대기 시간 $e$ 를 **설계 변수**로 올린 것이 이 논문의 기여다 — 기다릴수록 라벨은 정확해지고 데이터는 낡는다는 교환을 손잡이로 만들었다.

**FNW는 $e = 0$ 인 특수해다.** 창이 0이면 모든 전환이 창 밖 전환이라 $p_{dp} = p$, 관측 음성은 전체 클릭이라 $p_{rn} = 1 - p$. 대입하면 양성 $1+p$, 음성 $(1+p)(1-p)$ 로 FNW 가중치와 정확히 같아진다. 현행 `_fnw`가 이 지점이다.

**C1에 대한 대응** — 현행 온라인 3종은 $e = 0$ 이라 모든 클릭이 일단 오답으로 시작한다. 창을 1시간만 둬도 그 안에 오는 전환은 처음부터 양성으로 들어가 가짜 미전환 비율이 크게 떨어진다. 가중치가 보조 모델의 확률 두 개로만 만들어져, `_fnw`처럼 모델 자기 예측 $\hat p$ 를 되먹이는 구조도 사라진다. **배치가 아니라 온라인 라인을 고치는 카드**라 GDFM·ULC와 겹치지 않는다.

**C2에 대한 대응** — 없음. 중요도 가중은 이미 관측된 샘플 사이에서 무게를 재분배할 뿐이라, 재분배할 관측 양성이 없는 신규 그룹에는 닿지 않는다. 오히려 창만큼 데이터가 늦어져 약하게 역행한다.

**실험 결과** — Criteo·Taobao 스트리밍 평가. R-지표는 Vanilla = 0, Oracle = 1 기준으로 지연 피드백 간격을 얼마나 메웠는지.

| | Criteo AUC | Criteo NLL | Criteo R-NLL | Taobao AUC | Taobao NLL | Taobao R-NLL |
|---|---|---|---|---|---|---|
| Vanilla | 0.8376 | 0.4047 | 0 | 0.8842 | 0.1141 | 0 |
| FNW | 0.8373 | 0.4033 | 0.08 | 0.8845 | 0.1137 | 0.06 |
| **ES-DFM** | **0.8402** | **0.3924** | **0.68** | **0.8895** | **0.1112** | **0.47** |
| Oracle | 0.8450 | 0.3868 | 1 | 0.8949 | 0.1079 | 1 |

FNW는 NLL 간격을 거의 못 메우는데(6~8%) ES-DFM은 절반 이상 메운다. Chapelle DFM은 스트리밍에서 수렴하지 못했다(Criteo NLL 1.26). **온라인 A/B**(Taobao): AUC +0.3%, CVR +0.7%, GMV +1.8%.

### DEFER & DEFUSE — ES-DFM 후속

> **DEFER** (Gu et al., KDD 2021) — https://arxiv.org/abs/2104.14121  
> **DEFUSE** (Chen et al., WWW 2022) — https://arxiv.org/abs/2202.06472  
> 둘 다 Alibaba. ES-DFM이 남긴 문제를 하나씩 고친다.  

**DEFER — 진짜 음성도 다시 넣자.** 어트리뷰션 창이 지나면 전환한 클릭만 아니라 **모든 클릭**을 확정 라벨로 재투입한다. 전체 클릭이 두 번씩 들어가므로 $q(x) = p(x)$ 가 정확히 성립하고, 창 뒤에 돌아오는 미전환은 "아직 모름"이 아니라 **확실히 안 샀다**는 확정 정보가 된다. 관측 음성 가중치는 $q_{defer}(y{=}0 \mid x) = p(y{=}0 \mid x) + \tfrac12 f_{dp}(x)$ 에서 나온다. 창 길이를 상품별로 예측하는 확장도 제안. **비용은 데이터 2배**이고, 재투입까지 창만큼 기다리므로 창이 길면 확정 정보가 늦다.

세 방법을 같은 표기로 놓으면 차이가 보인다. 각 칸은 클릭 한 건이 스트림에 들어가는 라벨의 순서다(대기 창 1시간, 어트리뷰션 창 7일 기준).

| 클릭의 실제 결과 | FNW | ES-DFM | DEFER |
|---|---|---|---|
| 빨리 삼 (창 안 전환) | 0, 1 | 1 | 1, 1 |
| 늦게 삼 (창 뒤 전환) | 0, 1 | 0, 1 | 0, 1 |
| 안 삼 | 0 | 0 | 0, 0 |

FNW는 클릭 즉시 0, 전환 시 1이다. ES-DFM은 1시간 뒤 한 번 넣고 늦은 전환만 1을 추가한다. DEFER는 ES-DFM에 "안 삼"의 두 번째 0과 "빨리 삼"의 두 번째 1을 더한 것이다. 마지막 열에서 모든 행이 두 자리인 것이 "모든 클릭을 두 번 넣는다"의 뜻이고, 그래서 $q(x) = p(x)$ 가 된다. 1 → 0은 없다 — 어트리뷰션 창 안에서 전환은 취소되지 않는다(환불까지 다루는 것은 TESLA의 범위).

**대기 창과 어트리뷰션 창**

| | 누가 정하나 | 조정 가능? | 역할 |
|---|---|---|---|
| 대기 창 $w_1$ (예: 1시간) | 모델러 | 예, 튜닝 대상 | 클릭 후 이만큼 기다렸다 첫 라벨을 붙인다. ES-DFM의 $e$ 와 같은 것 |
| 어트리뷰션 창 (예: 7일) | 사업 규칙 | 아니오, 주어진 값 | 이 안의 전환만 성과로 인정·과금. 지나면 라벨이 확정된다 |

두 번째 방출이 어트리뷰션 창인 이유는 그때가 **라벨이 더 바뀌지 않는 최초 시점**이기 때문이다. DEFER의 확정 정보가 도착하는 시점은 이 창에 묶여 있어 앞당길 수 없다. 7일 목적에서는 7일 뒤, vcvr(1일)에서는 하루 뒤에 들어온다.

**관측 라벨 분포 $q(y{=}0 \mid x)$ — ES-DFM vs DEFER.** 클릭 10개, 빨리 삼 1 · 늦게 삼 1 · 안 삼 8 기준($p = 0.2$, $p_{dp} = 0.1$).

| | 스트림 구성 | 0인 행 | 전체 행 | 식 | 값 |
|---|---|---|---|---|---|
| ES-DFM | 1시간 뒤 라벨, 늦은 전환만 1 추가 | 안 삼 8 + 늦게 삼 첫 줄 1 = 9 | 10 + 1 = 11 | $\dfrac{(1-p) + p_{dp}}{1 + p_{dp}}$ | 0.818 |
| DEFER | 1시간 뒤 라벨, 7일 뒤 **모든** 클릭 확정 라벨 추가 | 안 삼 16 + 늦게 삼 첫 줄 1 = 17 | 20 | $(1-p) + \tfrac12 p_{dp}$ | 0.85 |
| 진짜 | | | | $1 - p$ | 0.8 |

- **분자** — 관측 시점에 0으로 보인 것. 진짜 미전환과 아직 안 온 전환이 섞여 있다. DEFER는 안 삼이 두 번 세어져 $2(1-p)$ 가 된다.
- **분모** — 전체 행 수. ES-DFM은 늦게 삼만 추가 행이라 $1 + p_{dp}$ 로 $x$ 에 따라 달라진다. DEFER는 모두 두 번이라 상수 2이고, 나누면 사라진다.
- 둘 다 진짜 0.8보다 크고, 그 차이를 되돌리는 것이 각 방법의 음성 가중치다.

**피처 분포 $q(x)$ 와 $p(x)$.** 중요도 가중치는 두 인수의 곱이다.

$$
\frac{p(x, y)}{q(x, y)} = \frac{p(x)}{q(x)} \times \frac{p(y \mid x)}{q(y \mid x)}
$$

- **뒤쪽** 라벨 왜곡은 셋 다 위 표의 $q(y \mid x)$ 로 정확히 보정한다.
- **앞쪽** $p(x)/q(x)$ 는 "어떤 클릭이 스트림에 몇 번 나오나"다. FNW·ES-DFM은 전환한 클릭만 두 번 들어가서 산 사람의 피처가 실제보다 자주 보이는데, 이 차이를 무시하고 $q(x) \approx p(x)$ 로 놓는다. 그만큼의 편향이 남는다.
- DEFER는 안 온 클릭에도 7일 뒤 확정 0을 넣어 모든 클릭을 두 번 만든다. 산 사람도 안 산 사람도 두 줄이라 등장 비율이 실제와 같아지고 $q(x) = p(x)$ 가 **가정이 아니라 사실**이 된다. 그 두 번째 0은 비율을 맞추는 동시에 "확실히 안 삼"이라는 확정 정보를 처음으로 모델에 넣는다.

**DEFUSE — 관측 음성을 둘로 쪼개자.** 관측 샘플을 IP(즉시 양성)·FN(가짜 음성)·RN(진짜 음성)·DP(지연 양성)로 나누고 종류별 가중치를 준다. 양성 쪽 IP·DP는 스트림에서 이미 구분되므로, 실제로 새로 쪼갠 것은 **관측 음성을 FN과 RN 둘로** 나눈 것이다. FN과 RN은 관측 시점에 똑같이 0으로 보이는데 기존 방법은 이 둘을 구분하지 않았고, 그것이 **불편성이 깨지는 지점**이라고 짚는다(중요도 샘플링은 라벨 불변을 가정하는데 지연 피드백은 같은 클릭의 라벨이 0→1로 바뀜). 해법은 숨은 변수 $z(x)$ = 관측 음성이 가짜 음성일 확률.

$$
\mathcal L_{ub} = \int \underbrace{q(x)\,dx}_{(1)}\ \sum_{v}\ \underbrace{q(v \mid x)}_{(2)}\ \underbrace{\frac{p(x)}{q(x)}}_{(3)}\ \underbrace{\frac{p(y(v,d) \mid x)}{q(v \mid x)}}_{(4)}\ \underbrace{\ell\big(x, y(v,d); f_\theta(x)\big)}_{(5)}
$$

스트림에서 피처 $x$ 가 나올 확률에 (1) 그 클릭에 관측 라벨 $v$ 가 찍힐 확률을 곱하고 (2), 피처 빈도를 보정하고 (3), 관측 라벨 대신 진짜 라벨 기준으로 바꿔주는 비율을 곱한 뒤 (4), 진짜 라벨 기준의 CE loss를 곱한다 (5).

**$w_i$ 는 $\mathcal L_{ub}$ 의 (3)×(4)다.**

$$
w_i(x) = \underbrace{\frac{p(x)}{q(x)}}_{(3)} \times \underbrace{\frac{p(y(v_i,d) \mid x)}{q(v_i \mid x)}}_{(4)} = \frac{p\big(x, y(v_i,d)\big)}{q(x, v_i)}
$$

조건부 확률의 곱이 결합확률이 되어 논문의 정의와 같아진다. 이걸로 $\mathcal L_{ub}$ 를 다시 쓰면 식 (14)다.

$$
\mathcal L_{ub} = \int q(x) \sum_{i \in \{IP, FN, RN, DP\}} q(v_i \mid x)\; w_i(x)\; \ell\big(x, y(v_i,d); f_\theta\big)
$$

바뀐 것은 $\sum_v$ 가 $\sum_i$ 로 바뀐 것 하나. $v{=}0$ 안에 FN과 RN이, $v{=}1$ 안에 IP와 DP가 있고, (4)의 분자 $p(y \mid x)$ 가 FN(진짜 1)과 RN(진짜 0)에서 다르니 가중치를 종류별로 따로 둬야 한다. 논문은 (3)은 여전히 1로 근사하므로 실제 계산되는 것은 (4)다.

**네 종류의 가중치** (대기 창 1시간, 어트리뷰션 창 7일 기준)

| 종류 | 이 클릭에 일어난 일 | 관측 $v$ | 진짜 $y$ | 가중치 | 손실 방향 |
|---|---|---|---|---|---|
| IP 즉시 양성 | 1시간 안에 샀다 | 1 | 1 | $1 + f_{dp}$ | 양성 |
| DP 지연 양성 | 1시간 뒤에 샀다. **복사본** 줄 | 1 | 1 | $1$ | 양성 |
| FN 가짜 음성 | 1시간 뒤에 샀다. **첫** 줄 | 0 | 1 | $f_{dp}$ | 양성 |
| RN 진짜 음성 | 끝까지 안 샀다 | 0 | 0 | $1 + f_{dp}$ | 음성 |

- **가중치의 의미** — 늦게 산 사람이 두 줄로 들어가 전체 행이 $1 + p_{dp}$ 배로 늘었으므로 모든 행의 비중이 그만큼 눌려 있다. 그래서 **모든 클릭이 한 클릭 몫 $1 + f_{dp}$ 를 받는다.** 한 줄짜리 IP·RN은 그 줄이 다 받고, 두 줄짜리 DP·FN은 합쳐서 받는다. 논문은 DP에 1, FN에 $f_{dp}$ 로 나눈다. 제약으로 쓰면 $w_{IP} = w_{RN} = 1 + f_{dp}$, $w_{DP} + w_{FN} = 1 + f_{dp}$.
- **손실 방향은 진짜 라벨 기준** — IP·DP·FN은 진짜가 1이라 양성 항으로, RN만 음성 항으로 간다. 기존 방법과의 차이는 FN 줄 하나다.
- **학습 시점에 구분되는 것** — IP vs DP는 구분된다(창 안에 왔으면 IP, 복사본이면 DP). FN vs RN은 **구분되지 않는다**(둘 다 0으로 찍혀 있고 결과를 모름). 그래서 관측 음성 하나를 $z$ 와 $1-z$ 로 쪼개 두 항에 동시에 넣는다.

**ES-DFM과의 차이 — 0으로 들어온 것을 어떻게 쓰나.** 클릭 10개(빨리 삼 1 · 늦게 삼 1 · 안 삼 8), 0으로 찍힌 행 9개 기준.

ES-DFM은 9개를 한 덩어리로 음성 항에 넣는다.

$$
(1 + f_{dp})\, p_{rn} \cdot \log(1 - f_\theta)
$$

$p_{rn} = 8/9$ 라 "9개 중 8개만큼만 음성으로 믿겠다"이고, 나머지 1개 몫은 버려진다. 늦게 산 사람의 첫 줄이 사실 양성이라는 정보는 쓰지 않고, 그 사람의 양성 몫은 며칠 뒤 도착하는 복사본 한 줄이 전부 담당한다.

DEFUSE는 9개 각각을 둘로 쪼갠다.

$$
\mathcal L_{neg} = z \cdot f_{dp} \cdot \log f_\theta \;+\; (1 - z)(1 + f_{dp}) \cdot \log(1 - f_\theta)
$$

뒤쪽은 $1 - z = p_{rn}$ 이라 ES-DFM과 같은 숫자다. **앞쪽이 새로 생긴 것** — ES-DFM이 버리던 $z = 1/9$ 몫을 "이건 사실 살 사람"이라며 양성 항으로 보낸다. 늦게 산 사람은 양성 신호를 복사본 한 줄에서만 받다가 이제 첫 줄의 $z$ 몫에서도 받고, 첫 줄이 복사본보다 며칠 먼저 도착하므로 **양성 정보가 더 일찍, 더 많이 들어온다.** 한 줄로: 기존에는 0으로 들어온 것을 전부 음성 항에 가중치를 줘서 넣었는데, DEFUSE는 그중 나중에 1로 바뀔 비율을 $z$ 로 추정해 그만큼을 양성 항에 넣는다.

**$z$ — 관측 음성이 사실 FN일 확률.**

$$
z(x) = \frac{p(y{=}1, d > w_o \mid x)}{p(y{=}0 \mid x) + p(y{=}1, d > w_o \mid x)} = 1 - p_{rn}(x)
$$

$z$ 를 직접 하지 않고 $f_{rn}$ 을 학습해 $z = 1 - f_{rn}(x)$ 로 쓴다 — ES-DFM의 진짜 음성 분류기 그대로. 확정된 과거 로그에서 관측 시점에 0으로 보였던 클릭만 골라, 끝까지 안 샀으면 1, 창 뒤에 샀으면 0으로 라벨을 붙인 이진 분류기다. 

**Bi-DEFUSE (4.2) — 창 안 전환은 보정 없이, 나머지는 DEFUSE로.** 창 안 전환(IP)은 관측 시점에 이미 정답을 아니 보정 없이 배우고, DP·FN·RN은 4.1의 DEFUSE로 배운다. 보정 오차를 창 뒤 헤드 하나에 가둬 분산을 줄인다.

헤드 둘, 출력 둘. 4.1은 헤드가 하나라 출력 $f_\theta$ 가 곧 pCVR이었다. Bi-DEFUSE는 답하는 질문이 다른 헤드 둘로 나눈다.

| | 질문 | 클릭 10개 예시 정답 |
|---|---|---|
| $f_{IP}(x)$ | 창 안에 살 확률 | 빨리 삼 1 → 0.1 |
| $f_{DP}(x)$ | 창 뒤에 살 확률 | 늦게 삼 1 → 0.1 |
| $f_\theta(x)$ (4.1) | 결국 살 확률 | 산 사람 2 → 0.2 |

$$
p(y{=}1 \mid x) = f_{IP}(x) + f_{DP}(x)
$$

4.1은 0.2를 통째로 배우고, Bi-DEFUSE는 0.1과 0.1을 따로 배워 더한다.

- **창 안 헤드 — 보정 없음.** $y_{IP}$ 는 "창 안에 샀으면 1, 아니면 0". 늦게 살 사람도 안 살 사람도 창 안에는 안 샀으니 0이 정답이다. 가짜 음성이 없어 진짜 분포 $p$ 에서 그대로 뽑는다.

$$
\mathcal L_{IP} = -\int p(x, y_{IP})\Big[\, y_{IP}\log f_{IP} + (1 - y_{IP})\log(1 - f_{IP}) \Big]
$$

- **창 뒤 헤드 — DEFUSE 가중치와 $z$.** $v_{DP}$ 는 복사본이면 1, 창 안에 안 온 첫 줄이면 0. 0 줄에 FN·RN이 섞여 있어 4.1 식 (15)를 그대로 쓴다. 다른 것은 IP 항이 빠지고 식 안의 $f_\theta$ 자리에 $f_{DP}$ 가 들어가는 것뿐이다.

$$
\mathcal L_{DP} = -\int q(x, v_{DP})\Big[\, v_{DP}\, w'_{DP}\log f_{DP} + (1 - v_{DP})\big( z'\,w'_{FN}\log f_{DP} + (1 - z')\,w'_{RN}\log(1 - f_{DP}) \big) \Big]
$$

$$
w'_{DP} + w'_{FN} = 1 + f_{dp}, \qquad w'_{RN} = 1 + f_{dp}, \qquad \mathcal L = \mathcal L_{IP} + \mathcal L_{DP}
$$

한 클릭이 두 헤드에 들어가는 방식:

| 클릭 | 창 안 헤드 $y_{IP}$ | 창 뒤 헤드 $v_{DP}$ |
|---|---|---|
| 빨리 삼 | 1 | 없음 |
| 늦게 삼 | 0 | 첫 줄 0 (FN 몫 $z'$), 복사본 1 (DP) |
| 안 삼 | 0 | 첫 줄 0 (RN 몫 $1 - z'$) |

- **왜 나누나.** 통째로 배우면 $z$·가중치의 보정 오차가 0.2 전체에 묻는데, 나누면 창 뒤 몫 0.1에만 묻는다. 창 안 몫이 클수록, 즉 어트리뷰션 창이 짧을수록 이득이 크다. 논문 실험에서 $w_a \le 7$ 일이면 Bi-DEFUSE가 DEFUSE보다 낫고 그 이상이면 뒤집힌다(Criteo-30d에서는 DEFUSE 52.3% vs Bi-DEFUSE 37.3%, Taobao에서는 55.1% vs 66.3%). 우리 7일 목적은 경계, 1일 목적 vcvr은 유리한 쪽.
- **구조** — MMoE. 창 안 전문가, 공유 전문가, 창 뒤 전문가를 게이트로 섞는다. 게이트를 빼거나 두 헤드를 독립 모델로 만들면 성능이 떨어졌다.

**관계** — ES-DFM(대기 창) → DEFER(재투입 대상 확대) → DEFUSE(관측 음성을 FN·RN으로 분해)의 단계적 업그레이드. 셋 모두 `_fnw` 가중치 자리에 들어가고 DEFER만 스트림 재투입 경로가 하나 더 필요하다. **DEFUSE의 $z(x)$ 는 ULC의 $w$ 와 같은 것** — ULC는 이 음성 항에서 중요도 가중치를 떼고 경과 시간을 조건에 넣은 배치판이다. **C2에는 둘 다 닿지 않는다** — DEFER 서론이 "새 캠페인 추가 등 분포 변화"를 동기로 들지만 해법은 관측 샘플 재가중이고 보조 모델은 과거 데이터로 학습된다.

**실험 결과 (DEFER)** — Criteo·Taobao-30d 스트리밍. RI는 Pre-trained = 0%, Oracle = 100%.

| | Criteo AUC | Criteo RI-NLL | Taobao AUC | Taobao RI-NLL |
|---|---|---|---|---|
| FNW | 0.8348 | 92.5% | 0.6440 | 87.8% |
| ES-DFM | 0.8373 | 95.8% | 0.6453 | 90.1% |
| **DEFER** | **0.8394** | **96.6%** | **0.6483** | **90.9%** |
| Oracle | 0.8429 | 100% | 0.6537 | 100% |

온라인 A/B: 스트리밍(장바구니, 1일 창) CVR +8.5%, 오프라인 멀티태스크 변형(구매, 7일 창) CVR +6%. 둘 다 주 트래픽 배포.

**실험 결과 (DEFUSE)** — Criteo-30d·1d, Taobao 스트리밍. RI는 Pre-trained = 0%, Oracle = 100%.

| | Criteo-30d RI-AUC | Criteo-1d RI-AUC | Taobao RI-AUC |
|---|---|---|---|
| FNW | 35.8% | 33.3% | 39.8% |
| ES-DFM | 46.1% | 92.1% | 52.0% |
| DEFER | 38.9% | 94.2% | 51.0% |
| **DEFUSE** | **52.3%** | **95.2%** | 55.1% |
| Bi-DEFUSE | 37.3% | 96.3% | **66.3%** |

온라인 A/B(30분 관측 창, 1일 어트리뷰션): CVR +2.28%. 같은 FNW가 DEFER 표에서는 RI-NLL 90%대, DEFUSE 표에서는 RI-AUC 35%다 — 기준선·전처리가 달라 **논문 간 수치 비교는 불가**, 표 안 상대 위치만 읽는다.

**실험 설계 함의** — 셋이 "클릭 후 $e$ 시간 기다렸다 emit하는 스트림"을 공유하므로, 오프라인 스트리밍 재생으로 대조군(현행 `_fnw`)·ES-DFM·+DEFER·+DEFUSE·+둘 다를 **한 번에 비교**하고 승자만 온라인 A/B로 올린다. DEFER가 7일 창에서 이득인지는 이 비교로만 답이 나온다. DEFUSE 팔의 결과는 배치 라인 ULC의 사전 검증이 된다.

### GDFM — post-click 행동을 전환의 대리 신호로

> **GDFM** (Yang & Zhan, NeurIPS 2022) — https://arxiv.org/abs/2206.00407
> C1과 C2를 한 틀에서 다루는 유일한 논문. 아래는 그 요지와 두 원인에 대한 대응.

#### 요지

유저 트렌드나 프로모션 등에 의해 현재 시점의 클릭-전환 분포 $p_t(y \mid x)$ 는 매 순간 불안정하게 변화한다. 반면 **전환할 유저 $y$ 가 장바구니에 담는 행동 $a$ 를 할 조건부 확률** $p(a \mid x, y)$ 는 시간에 거의 구애받지 않고 안정적으로 유지된다.

이 가정 아래, 며칠 뒤에나 모이는 진짜 전환 라벨을 기다리는 대신 10~30분이면 모이는 행동 데이터 $p_t(a \mid x)$ 와 이 고정된 연결고리를 이용해 현재 시점의 전환율 $p_t(y \mid x)$ 를 **전환 라벨 대비 훨씬 짧은 지연으로 간접 추정**한다.

그리고 각 행동 신호에 **정보량**과 **신선도** 두 가중치를 곱해 비중을 매긴다. 늦게 얻을수록 정보량은 늘고 신선도는 떨어지므로 둘의 곱이 최대가 되는 지점이 생긴다. 이 비중으로 손실 함수를 업데이트한다.

$$w_j = \underbrace{e^{-\alpha H(y \mid a_j)}}_{\text{정보량}} \cdot \underbrace{e^{-\beta \delta_j}}_{\text{신선도}}$$

여기에 확정 라벨로 학습한 모델을 KL 앵커로 붙잡아, 정보 없는 행동을 넣어도 성능이 떨어지지 않도록 보장한다. 운영에서 새 신호를 안전하게 추가하기 위한 장치다.

#### C1에 대한 대응

C1의 구조는 "신규 그룹은 클릭이 전부 최근이라 미전환이 할인되고, 자기 데이터가 기울기를 못 만들어 예측이 공유 파라미터 값에 머문다"였다.

GDFM은 **배우는 대상을 바꿔** 이 구조를 빠져나간다. 10분 뒤 조회한 장바구니는 "아직 모름"이 아니라 **그 시점에 값이 확정된 관측**이다. 전환 라벨은 검열돼 있지만 행동은 검열돼 있지 않다. 할인 없이 온전한 기울기가 그룹 임베딩으로 들어가고, 침묵 구간이 며칠에서 분 단위로 줄어든다.

가짜 미전환을 만들지 않는 점도 여기에 닿는다. FNW·ES-DFM은 틀린 라벨을 붙였다가 가중치로 되돌리지만 GDFM에는 그 단계가 없다. Taobao 실험에서 두 방법의 NLL이 사전학습 모델보다 크게 나빴던 것이 그 대가였다(FNW −361%, ES-DFM −214%, GDFM +49.6%). **AUC는 유지되는데 NLL만 무너지는 패턴** — 우리가 AUC 대신 RIG를 보는 이유와 같은 지점이고, 현행 온라인 3종이 FNW 계열이다.

#### C2에 대한 대응

**C2-② 낡은 prior** — 신선도 가중이 오래된 정보를 지수적으로 깎는다. 신규 그룹이 받던 "지난 30일 평균"의 발언권이 줄고 최근 신호의 발언권이 커진다. 이 격차를 temporal gap으로 따로 이름 붙여 정식화한 논문은 GDFM뿐이다.

**C2-① 표본 부족** — 교환이다. 신규 그룹은 몇 시간 동안 전환이 0건일 수 있지만 장바구니는 구매보다 자주 일어나 쓸 수 있는 신호 건수가 늘어난다. 대신 신호 하나당 정보량은 전환 라벨보다 적다(sampling gap). GDFM은 이 교환을 없애주지 않고 **얼마로 쳐야 하는지 계산해준다.**

**실험 결과** — Criteo·Taobao 스트리밍 평가. Pretrain = 0%, Oracle = 100% 기준 상대 개선율.

| | Criteo AUC | Criteo NLL | Taobao AUC | Taobao NLL |
|---|---|---|---|---|
| FNW | 62.0% | 40.3% | 39.6% | −361% |
| ES-DFM | 71.4% | 66.2% | 62.6% | −214% |
| **GDFM** | **74.9%** | **72.4%** | **79.4%** | **+49.6%** |

Taobao에서 FNW·ES-DFM의 NLL이 사전학습 모델보다 나쁘다 — 가짜 미전환의 대가. Criteo에는 post-click 행동 로그가 없는데도 GDFM이 앞섰다(조기 전환을 조회 시각별로 쪼개 가중). Ablation(Criteo): 정보량 가중 $\alpha$ 제거 시 하락 큼, 신선도 가중 $\beta$ 제거는 영향 작음. 무정보 행동을 추가해도 성능 유지.

> **한계**  
① 조회 시각 $\delta_j$ 만큼의 지연은 남는다지연을 없애는 게 아니라 며칠에서 분 단위로 옮기는 것.  
② 행동이 무정보면 성능이 안 떨어질 뿐 이득도 없다 — **우리 로그에서 장바구니·구매 간 조건부 엔트로피를 먼저 재야 판단 가능.** 

### ULC — 가중치 대신 라벨을 고친다

> **ULC** (Wang et al., KDD 2023) — https://arxiv.org/abs/2307.12756

미전환 샘플마다 **전환 확률 라벨** $w_i = P(c_i = 1 \mid x_i, e_i, v_i = 0)$ 을 붙여 손실을 세 항으로 구성한다. 관측 시점 $T$ 이전에 전환이 확인된 샘플 항, 그리고 관측상 미전환인 샘플을 $w_i$ 만큼 전환자로 센 항과 $1 - w_i$ 만큼 진짜 미전환자로 센 항이다. 뒤의 두 항은 **같은 샘플을 확률로 쪼갠 것**이다. $w_i$ 가 정확하면 진짜 라벨로 학습한 손실과 기댓값이 같다(Theorem 1).

$$\mathcal{L}_{LC} = \frac{1}{|D|}\sum_i \big[\, v_i \log f + w_i(1-v_i)\log f + (1-w_i)(1-v_i)\log(1-f) \,\big]$$

$w$ 를 학습할 데이터는 원본에 없으므로 **counterfactual labeling**으로 만든다. 실제 데이터 수집 마감보다 $\tau$ 만큼 앞선 가상의 마감을 두고, 그 시점에 미전환이던 클릭 중 두 마감 사이에 전환한 것을 $w = 1$, 나머지를 $w = 0$ 으로 라벨링한다. 보조 모델은 이 데이터로 평범한 이진 분류 학습을 하고, 추론 시 연속 확률 $w$ 를 내놓는다. 보조 모델의 임베딩은 CVR 모델에서 복사해 초기화하며, 논문 실험에서 **교대 학습 1회로 이득의 대부분**이 나왔다.

**FNW와의 관계** — 둘 다 "아직 전환이 안 온 클릭은 완전한 0도 1도 아니다"에서 출발하는 soft label 계열이다. 차이는 둘이다. ① FNW는 음성 줄과 미래의 양성 복사본 **두 줄에 나눠** 적고, ULC는 한 줄에 $w : 1-w$ 로 적는다(샘플당 질량 합 = 1). ② 숫자의 출처가 다르다 — FNW는 모델 자기 예측 $\hat p = P(c=1 \mid x)$ 로 **경과 시간을 보지 않고**, ULC는 별도 모델이 $e$ 와 "지금까지 미전환"을 조건으로 받아 계산한다.

**C1에 대한 대응** — 배치 DFM의 감쇠는 "아직 모름"의 **영향력을 깎는** 방식이라 기울기까지 함께 사라진다. ULC는 영향력을 깎지 않고 **값을 고친다.** 이상적 $w$ 아래 편향 없음이 보장된다.

**C2에 대한 대응** — 재가중은 가짜 미전환과 비슷한 **관측 양성의 무게를 올려** 간접 보충하는데, 신선한 가짜 미전환에는 비슷한 관측 양성이 없을 수 있다(위 C2 문제점 지적의 인용). 라벨 교정은 그 그룹 **자신의** 미전환 샘플을 직접 고치므로 과거 양성에 기대지 않는다.

**실험 결과** — Criteo 오프라인(3주 학습), MLP 백본. RI는 Vanilla = 0, Oracle = 1.

| | AUC | RI-AUC | LL | RI-LL |
|---|---|---|---|---|
| Vanilla | 0.8208 | 0 | 0.4505 | 0 |
| DFM | 0.8264 | 0.24 | 0.4378 | 0.26 |
| FSIW | 0.8335 | 0.55 | 0.4178 | 0.68 |
| nnDF | 0.6859 | −5.79 | 0.5969 | −3.05 |
| **ULC** | **0.8403** | **0.84** | **0.4104** | **0.84** |
| Oracle | 0.8441 | 1 | 0.4025 | 1 |

백본 4종 평균으로 Vanilla–Oracle 간격을 AUC 77.6%, PRAUC 66.6%, LL 83.1% 메웠다. nnDF는 전역 의존성 때문에 미니배치 최적화가 불가능해 Vanilla보다 크게 나빴다 — **우리 학습 구조에서는 사실상 후보가 아니다.** $\tau$ 최적은 약 1주, 교대 학습은 1회로 충분. 사내 게임 광고 데이터에서도 DFM·FSIW를 앞섰다(수치 미공개, 그림만).

> **한계**  
> ① 보조 모델 자신도 지연 피드백을 겪는다 — $\tau$ 보다 늦게 전환하는 샘플은 음성으로 잘못 학습되며, 논문은 이를 향후 과제로 남긴다.  
> ② $\tau$ 에 최적점이 있다(Criteo 약 1주). 짧으면 오라벨이 늘고, 길면 보조 모델 학습 데이터가 낡아 **신선한** 가짜 미전환을 못 고친다.  
> ③ 실험이 전부 오프라인 배치 설정이다. **GDFM은 스트리밍, ULC는 배치** — 우리 쪽에서도 ULC는 배치 DFM 라인, GDFM은 온라인 라인에 대응한다.  
> ④ 공동 학습은 오히려 나빠졌고(초반의 부정확한 보조 모델이 CVR 모델을 오도), CVR 예측값으로 잠재 양성을 찾는 전략도 효과가 없었다(진짜 음성 대비 1:50, 노이즈만 유입).

### 역할 분담

| | C1 | C2 |
|---|---|---|
| ES-DFM | 온라인 라인의 직접 해결책 | 해당 없음, 약하게 역행 |
| GDFM | 대기 시간을 분 단위로 단축 | 신선도 가중으로 정면 대응 |
| ULC | 미전환에 확률값 부여 | 신규 그룹 자기 데이터로 보정 |

## C3 — 옵티마이저 설정

> 튜닝 예
> | 항목 | 현재 값 | 실험 방향 | 근거 |
> |---|---|---|---|
> | 온라인 APP 임베딩 학습률 | adam 1e-4, dense와 동일 | 임베딩만 param group 분리해 1e-3 | Adam은 스텝 크기 ≈ 학습률. $10^{-5}$ 에서 의미 있는 크기까지 수백~수천 스텝. PF·MEM은 이미 1e-3으로 운영 중 |
> | 온라인 APP weight decay | 1e-5 | 임베딩은 0 | 매 스텝 임베딩을 0으로 당김. 새 행은 커져야 하는데 반대 방향. PF·MEM은 0 |
> | 배치 Clippy `lambda_abs` | 0.01, 코드 하드코딩 | config로 꺼내 0.05 | 클립 상한 $= 0.5\lvert w\rvert + 0.01$. $\lvert w\rvert \approx 0$ 인 새 행은 0.01에 고정. 클립 계수가 텐서 단위라 새 행이 테이블 전체를 늦출 수 있음 |
> | 배치 Clippy `lambda_rel` | 0.5 | 유지 | 새 행에는 $\lvert w\rvert \approx 0$ 이라 영향 없음. 성숙 행 안정성 담당 |
> | 배치 Adagrad 누적기 | run마다 리셋 | 유지 | 리셋 직후 스텝이 크게 나가 새 행에 유리 |
> | 배치 크기 (배치 DFM) | 8192 | 조건부 4096 | 희소 행 기울기 ∝ 등장 횟수 / 배치 크기. Adagrad는 기울기 크기가 스텝에 반영. 처리량 비용 있음 |
> | `lr_scheduler` | false | 유지 | 켜도 상수 0.95를 곱할 뿐 스케줄이 아님 |

## C4 — group_id 임베딩 초기화

**의미 있는 값으로 초기화하면 초기 pCVR 안정에 도움이 될 수 있다.** 초기값 후보는 아래 순서로 본다.

| | 초기값 | 평가 |
|---|---|---|
| 1 | 전역 group_id 임베딩 평균 | 사실상 무효 — 지금도 예측이 "평균적인 그룹" 값이라 달라지는 게 없다 |
| 2 | **같은 계정·캠페인 형제 그룹 임베딩의 평균** | 싸고 정보가 있다. 형제가 없으면 계정 → 목적 순으로 폴백. **첫 실험감** |
| 3 | 속성(계정·캠페인·목적·소재)을 받아 임베딩을 생성하는 모델 | MetaEmb 방식. 가장 강하지만 학습 파이프라인이 하나 늘어난다 |

**구현 시 주의 두 가지.** ① 초기화 지점이 한 곳이 아니다 — 모델 생성 시점의 `torch.nn.init.normal_` 외에 `utils_online/export.py`의 슬롯 재사용 경로(`reset not used embedding`)와 `cmd/reset_embed.py`도 같이 고쳐야 한다. 안 그러면 은퇴한 슬롯을 물려받은 새 그룹은 여전히 $10^{-5}$ 로 시작한다. ② 옵티마이저 모멘트를 0으로 민 상태에서 출발값만 커지면 초반 갱신 동역학이 바뀐다.

**측정**: 초기화 방식별로 그룹 생성 후 경과 시간별 캘리브레이션 오차 곡선을 겹쳐 그린다. 첫 몇 시간 구간의 차이가 C4의 크기다.

# 결론 — 메인 / 옵션 / 참고

> 대상: `config/cvr/dfm`(배치 DFM, PUR)과 `config/cvr/dfom`(온라인 DFOM, PF·APP·MEM)을 쓰는 CVR 모델 6종. 신규 그룹의 캘리브레이션 도달 시간 $N$ 단축이 목적이고, 성숙 구간의 캘리브레이션·RIG 비열위가 제약이다.

| 위상 | 할 것 | 원인 | 적용 라인 | 근거 |
|---|---|---|---|---|
| 메인 1 | ES-DFM 도입 + DEFER·DEFUSE 확장 비교 | C1 | 온라인 PF·APP·MEM | Tier 1, Alibaba 배포 |
| 메인 2 | 옵티마이저 설정 조정 | C3 | 양쪽 | 논문 없음, 코드 확인 |
| 옵션 1 | group_id 임베딩 초기화 | C4 | 양쪽 | 콜드스타트 문헌(지연 피드백 밖). **구조 확인 필요** |
| 옵션 2 | GDFM | C1·C2 | 온라인 | Tier 2. **$H(y \mid \text{cart})$ 확인 필요** |
| 참고 | ULC | C1·C2 | 배치 DFM | Tier 2. DEFUSE와 같은 아이디어의 배치판 |

메인은 C1·C3만 다룬다. 신규 그룹의 첫 몇 시간을 겨냥하는 C2·C4는 옵션에 있다 — 메인의 근거는 직접 관련성이 아니라 **증거 강도와 공수**다.

## 0. 측정 먼저

$N$ 곡선, 신규 그룹의 첫 24시간 갱신 스텝 수·임베딩 노름 궤적, $H(y \mid \text{cart})$ 를 잰다. 아직 아무것도 재지 않았다. 스텝 수가 병목이면 C3가 아니라 C2-①이고, $H(y \mid \text{cart})$ 가 높으면 옵션 2는 접는다.

## 메인 1. ES-DFM 도입 + DEFER·DEFUSE 확장 비교 실험 (온라인, C1)

현행 `_fnw`는 대기 시간 $e = 0$ 인 특수해다. 스트림 emit을 $e$ 시간 뒤로 미루고 $p_{dp}$·$p_{rn}$ 보조 헤드를 붙여, `_fnw` 가중치 $(1+\hat p)$, $(1-\hat p)(1+\hat p)$ 를 $(1+p_{dp})$, $(1+p_{dp})\,p_{rn}$ 으로 바꾼다. 가짜 미전환이 줄고 자기 예측 되먹임이 사라진다. 비용은 sequoia 스트림 emit 시점 변경. C2에는 닿지 않는다.

DEFER(어트리뷰션 창 뒤 전체 클릭 재투입)와 DEFUSE(관측 음성 안의 가짜 음성 확률 $z(x)$)는 ES-DFM의 대기 창·보조 헤드를 그대로 깔고 각각 재투입 경로와 $z$ 추정을 더한다. 셋이 같은 스트림을 쓰므로 **오프라인 스트리밍 재생으로 대조군(현행 `_fnw`)·ES-DFM·+DEFER·+DEFUSE·+둘 다를 한 번에 비교**하고 승자만 온라인 A/B로 올린다. DEFER가 7일 창에서 이득인지는 이 비교로만 답이 나온다(논문: 1일 창에서 우세, 30일 창에서 열세). 로직·실험 수치는 위 소결 "DEFER·DEFUSE" 절.

## 메인 2. 옵티마이저 설정 조정 (양쪽, C3)

한 스텝의 갱신 크기를 정하는 것은 학습률·weight decay·Clippy 클립 상한·Adagrad 누적기 상태로, 전부 옵티마이저 영역이다. 예외는 **배치 크기** — 기울기가 배치 평균이라 희소한 그룹의 임베딩 행이 받는 기울기는 등장 횟수 / 배치 크기이고, Adagrad 계열은 그 크기가 스텝에 그대로 들어간다(Adam은 정규화하므로 온라인은 해당 없음). `epoches`는 스텝 수를 바꾸는 것이라 C3가 아니라 C2-① 쪽이다.

신규 그룹 임베딩의 이동 거리는 **스텝 수 × 스텝 크기**다. 스텝 수는 C2-①(표본), 스텝 크기가 C3다. **아래 변경은 성숙 그룹에도 같이 걸리므로 제약 지표(캘리브레이션·RIG)를 함께 본다.**

| 항목 | 현재 값 | 실험 방향 | 근거 |
|---|---|---|---|
| 온라인 APP 임베딩 학습률 | adam 1e-4, dense와 동일 | 임베딩만 param group 분리해 1e-3 | Adam은 스텝 크기 ≈ 학습률. $10^{-5}$ 에서 의미 있는 크기까지 수백~수천 스텝. PF·MEM은 이미 1e-3으로 운영 중 |
| 온라인 APP weight decay | 1e-5 | 임베딩은 0 | 매 스텝 임베딩을 0으로 당김. 새 행은 커져야 하는데 반대 방향. PF·MEM은 0 |
| 배치 Clippy `lambda_abs` | 0.01, 코드 하드코딩 | config로 꺼내 0.05 | 클립 상한 $= 0.5\lvert w\rvert + 0.01$. $\lvert w\rvert \approx 0$ 인 새 행은 0.01에 고정. 클립 계수가 텐서 단위 스칼라라 새 행이 group_id 테이블 전체를 늦출 수 있음 |
| 배치 Clippy `lambda_rel` | 0.5 | 유지 | 새 행에는 $\lvert w\rvert \approx 0$ 이라 영향 없음. 성숙 행 안정성 담당 |
| 배치 Adagrad 누적기 | run마다 리셋 (가중치만 상속) | 유지 | 리셋 직후 스텝이 크게 나가 새 행에 유리 |
| 배치 크기 (배치 DFM) | 8192 | 조건부 4096 | 희소 행 기울기 ∝ 등장 횟수 / 배치 크기. 처리량 비용 있음 |
| `lr_scheduler` | false | 유지 | 켜도 코드가 상수 0.95를 곱할 뿐 스케줄이 아님 |
| `forget` | 1.0 | 유지 | 테이블 리사이즈 때 기존 행 상속 비율. 새 행과 무관 |

**측정 먼저**: 신규 그룹이 첫 24시간에 받는 갱신 스텝 수와 임베딩 노름 궤적을 뽑는다. 스텝 수가 병목이면 C3가 아니라 C2-①이다.

## 옵션 1. group_id 임베딩 초기화 (양쪽, C4)

**선행 확인**: 현재 임베딩 테이블 구조에서 신규 행을 형제 그룹 평균으로 채우는 것이 가능한지 — 새 group_id가 인덱스를 받는 시점, 그 시점에 계정·캠페인 정보를 조회할 수 있는지, 슬롯 재사용 경로와 충돌하지 않는지.

현행 $N(0, 10^{-5})$ 대신 **같은 계정·캠페인 형제 그룹 임베딩의 평균**으로 초기화한다. 형제가 없으면 계정 → 목적 순 폴백. 전역 평균은 지금 예측과 같아 무효. 수정 지점은 `feature.py` 생성 시점, `export.py` 슬롯 재사용, `reset_embed.py` 세 곳.

C3와 결합 효과가 있다. Clippy 상대 클립이 $\lvert w\rvert$ 에 비례하므로 초기값이 학습된 크기(예: 0.1)면 스텝 상한이 $0.5 \times 0.1 = 0.05$ 로 `lambda_abs`의 다섯 배가 된다. **초기값을 키우는 것만으로 배치 라인의 C3가 함께 풀린다.**

## 옵션 2. GDFM — $H(y \mid \text{cart})$ 확인 후 (온라인, C1·C2)

**GDFM** — `OBJECTIVE_MAP`의 CART를 구매 헤드의 대리 신호로 연결한다. 현재 `mtldfm_v2`는 positive가 자기 action_type 헤드만 갱신해 장바구니가 구매 예측에 닿지 않는다. 신선도·정보량 가중은 로그 집계로 계산하고, KL 앵커는 확정 라벨로 도는 배치 DFM을 쓴다. **선행 조건**: 우리 로그에서 장바구니·구매 간 조건부 엔트로피 $H(y \mid \text{cart})$ 를 먼저 잰다. 무정보면 이득이 없다.

Tier 2라 A/B 없이 반영하지 않는다.

## 참고. ULC (배치, C1·C2)

**ULC** — delay 헤드 자리에 $w$ 헤드를 추가하고 counterfactual labeling($\tau$ 약 1주)으로 배치 잡 하나를 만든다. 검열항 대신 LC 손실을 쓰고 교대 학습 1회. 배치 DFM의 "아직 모름" 감쇠를 값 교정으로 대체한다. LC 손실은 계수가 모두 음이 아니라 nnDF식 방어가 필요 없다.

DEFUSE의 $z(x)$ 와 같은 아이디어의 배치판이다(중요도 가중치를 떼고 경과 시간을 조건에 넣음). 메인 1의 DEFUSE 팔 결과가 좋으면 배치 DFM에 올릴 근거가 되고, 나쁘면 우선순위를 내린다. Tier 2라 A/B 없이 반영하지 않는다.

## 순서

| | 항목 | 이유 |
|---|---|---|
| 0 | $N$ 곡선, 스텝 수·노름 궤적, $H(y \mid \text{cart})$ 측정 | 아직 아무것도 재지 않았다 |
| 1 | 메인 2 옵티마이저 조정 | config만. 학습 파이프라인 불변 |
| 2 | 메인 1 ES-DFM + DEFER·DEFUSE 비교 | 스트림 변경 1회. Tier 1 |
| 옵션 | 초기화 (구조 확인 후) · GDFM ($H$ 확인 후) | 선행 확인 결과에 따라 |
| 참고 | ULC | 메인 1의 DEFUSE 팔 결과에 따라 |

# Appendix

## Alibaba 3종(ES-DFM·DEFER·DEFUSE)의 인용 현황

> 2026-09-17 OpenAlex 기준. OpenAlex는 Google Scholar보다 적게 잡히므로 절대값보다 추세를 본다. 인용 논문의 arXiv가 확인된 경우 arXiv, 아니면 DOI로 연결했다.

### 연도별 인용 수

| | 총 인용 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| ES-DFM (AAAI 2021) | 30 | 3 | 3 | 10 | 3 | 4 | 7 |
| DEFER (KDD 2021) | 30 | — | 5 | 8 | 4 | 4 | 9 |
| DEFUSE (WWW 2022) | 27 | — | 2 | 9 | 4 | 4 | 8 |

**읽을 점 셋.** ① 세 편 모두 2026년이 2023년 다음으로 높거나 최고치다 — 줄지 않는다. ② 역할은 거의 전부 **베이스라인**이다. 자기 방법을 제안하고 셋을 비교 대상으로 놓아 이긴다. ③ 2025~2026 인용의 절반이 Alibaba 자신이고, 2026 TESLA는 CVR 타워 디바이어스에 ES-DFM 가중치 $w^+ = 1 + p_v\,p(h_v > W \mid y{=}1, x)$ 를 그대로 쓴다(부록 C 제목이 "Debiasing Strategy in ES-DFM"). 외부 기업이 셋을 프로덕션에 그대로 쓴다고 밝힌 사례는 없고, Kuaishou(TWICE)는 셋을 이기고 자기 방법을 배포했다.

### 인용 논문 목록

중복 제거 후 44편 + OpenAlex 미반영 2편. "인용" 열은 셋 중 어느 것을 인용했는지, "성격" 열은 그 논문에서 셋이 어떤 역할인지다.

| 연도 | 논문 | 소속 | 인용 | 성격 | 링크 |
|---|---|---|---|---|---|
| 2021 | Co-Transport for Class-Incremental Learning | Nanjing University | ES-DFM | 타 주제 인용 | [DOI](https://doi.org/10.1145/3474085.3475306) |
| 2021 | Conversion Prediction with Delayed Feedback: A Multi-task Learning Approach | Alibaba Group (China), University of Tennessee at Knoxville | ES-DFM | 지연 방법 (MM-DFM) | [DOI](https://doi.org/10.1109/icdm51629.2021.00029) |
| 2021 | Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeli | — | ES-DFM | 후속 방법 (DEFER) | [arXiv 2104.14121](https://arxiv.org/abs/2104.14121) |
| 2022 | Asymptotically Unbiased Estimation for Delayed Feedback Modeling via Label Correction | Alibaba Group (China) | DEFER, ES-DFM | 후속 방법 (DEFUSE) | [arXiv 2202.06472](https://arxiv.org/abs/2202.06472) |
| 2022 | Calibrated Conversion Rate Prediction via Knowledge Distillation under Delayed Feedback in | Chinese Academy of Sciences, Institute of Computing Technolo | DEFUSE, ES-DFM | 지연 방법 | [DOI](https://doi.org/10.1145/3511808.3557557) |
| 2022 | Cross-domain Recommendation via Adversarial Adaptation | Tencent (China) | DEFER | 타 주제 인용 | [DOI](https://doi.org/10.1145/3511808.3557277) |
| 2022 | KEEP: An Industrial Pre-Training Framework for Online Recommendation via Knowledge Extract | Alibaba Group (China), Tsinghua University | DEFER | 타 주제 인용 | [DOI](https://doi.org/10.1145/3511808.3557106) |
| 2022 | Learning Classifiers under Delayed Feedback with a Time Window Assumption | CyberAgent (Japan) | DEFER, DEFUSE, ES-DFM | 지연 방법 (nnDF) | [arXiv 2009.13092](https://arxiv.org/abs/2009.13092) |
| 2022 | Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models | Alibaba Group (China), Nanjing University | DEFER | 타 주제 인용 | [DOI](https://doi.org/10.1145/3511808.3557479) |
| 2023 | 3MN: Three Meta Networks for Multi-Scenario and Multi-Task Learning in Online Advertising  | Tencent (China) | DEFUSE | 타 주제 인용 | [DOI](https://doi.org/10.1145/3583780.3614651) |
| 2023 | Capturing Conversion Rate Fluctuation during Sales Promotions: A Novel Historical Data Reu | Alibaba Group (China), Nanjing University, University of Sci | DEFER, DEFUSE, ES-DFM | 응용 — 프로모션 CVR (HiFI) | [DOI](https://doi.org/10.1145/3580305.3599788) |
| 2023 | CollabEquality: A Crowd-AI Collaborative Learning Framework to Address Class-wise Inequali | University of Illinois Urbana-Champaign | DEFUSE | 타 주제 인용 | [DOI](https://doi.org/10.1145/3543507.3583871) |
| 2023 | Cross-domain Recommendation via Dual Adversarial Adaptation | Tongji University, University of Electronic Science and Tech | DEFER | 타 주제 인용 | [DOI](https://doi.org/10.1145/3632524) |
| 2023 | Dually Enhanced Delayed Feedback Modeling for Streaming Conversion Rate Prediction | Renmin University of China | DEFER, DEFUSE, ES-DFM | 지연 방법 (DDFM) — 셋을 베이스라인으로 비교 | [DOI](https://doi.org/10.1145/3583780.3614856) |
| 2023 | Entire Space Cascade Delayed Feedback Modeling for Effective Conversion Rate Prediction | Alibaba Group (China), Shandong University | DEFER, DEFUSE, ES-DFM | 지연 방법 (ESDF) | [arXiv 2308.04768](https://arxiv.org/abs/2308.04768) |
| 2023 | Freshness or Accuracy, Why Not Both? Addressing Delayed Feedback via Dynamic Graph Neural  | Zhejiang University of Science and Technology | DEFER, DEFUSE, ES-DFM | 지연 방법 | [DOI](https://doi.org/10.1109/icws60048.2023.00059) |
| 2023 | Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model | Alibaba Group (China) | DEFER | 타 주제 — 캘리브레이션 | [DOI](https://doi.org/10.1145/3580305.3599851) |
| 2023 | Leveraging Post-Click User Behaviors for Calibrated Conversion Rate Prediction Under Delay | Institute of Computing Technology, University of Chinese Aca | ES-DFM | 지연 방법 — post-click | [DOI](https://doi.org/10.1145/3583780.3615161) |
| 2023 | Modelling Delayed Redemption with Importance Sampling and Pre-Redemption Engagement | — | DEFUSE, ES-DFM | 응용 — 쿠폰 리뎀션 | [DOI](https://doi.org/10.1145/3580305.3599867) |
| 2023 | Online Conversion Rate Prediction via Neural Satellite Networks in Delayed Feedback Advert | Chinese Academy of Sciences, Institute of Computing Technolo | DEFER, DEFUSE, ES-DFM | 지연 방법 (NSN) | [DOI](https://doi.org/10.1145/3539618.3591747) |
| 2023 | RLTP: Reinforcement Learning to Pace for Delayed Impression Modeling in Preloaded Ads | Alibaba Group (China) | ES-DFM | 응용 — 노출 pacing | [DOI](https://doi.org/10.1145/3580305.3599900) |
| 2023 | Unbiased Delayed Feedback Label Correction for Conversion Rate Prediction | Huawei Technologies (China), Tsinghua University | DEFER, DEFUSE, ES-DFM | 지연 방법 (ULC) | [arXiv 2307.12756](https://arxiv.org/abs/2307.12756) |
| 2023 | Understanding Elapsed-time Sampling Delayed Feedback | Irvine University, Kindred Hospital Rancho, Lancaster Univer | ES-DFM | ES-DFM 분석 | [DOI](https://doi.org/10.4108/eai.2-6-2023.2334607) |
| 2024 | Addressing Delayed Feedback in Conversion Rate Prediction: A Domain Adaptation Approach | Duke University, Rice University, Samsung (United States) | DEFER, DEFUSE, ES-DFM | 지연 방법 — 도메인 적응 | [DOI](https://doi.org/10.1109/icdm59182.2024.00115) |
| 2024 | Calibration-compatible Listwise Distillation of Privileged Features for CTR Prediction | Alibaba Group (China), Shandong University | DEFER | 타 주제 인용 | [DOI](https://doi.org/10.1145/3616855.3635810) |
| 2024 | Debiasing the Conversion Rate Prediction Model in the Presence of Delayed Implicit Feedbac | Peking University, Peking University International Hospital | DEFER, DEFUSE, ES-DFM | 지연 방법 | [DOI](https://doi.org/10.3390/e26090792) |
| 2024 | Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approach | Alibaba Group (China) | DEFER | 타 주제 인용 | [DOI](https://doi.org/10.1145/3627673.3680068) |
| 2024 | Modeling User Attention in Music Recommendation | Huawei Technologies (China), Renmin University of China | DEFUSE | 타 주제 인용 | [DOI](https://doi.org/10.1109/icde60146.2024.00064) |
| 2024 | Online Conversion Rate Prediction via Multi-Interval Screening and Synthesizing under Dela | Institute of Computing Technology | DEFUSE, ES-DFM | 지연 방법 (MISS) | [DOI](https://doi.org/10.1609/aaai.v38i8.28726) |
| 2025 | Consumer Conversion Prediction Via Heterogeneous Graph Networks and Sparse Attention Learn | Yantai Academy of Agricultural Sciences | ES-DFM | 타 주제 인용 | [DOI](https://doi.org/10.1109/eiecc67963.2025.11409614) |
| 2025 | Mind the Gap: Delayed Label Bias-Variance Tradeoffs in Predicting Likelihood of Nonpayment | Meta (United States), Northeastern University | DEFER, DEFUSE | 응용 — 미납 예측, post-transaction pseudo-label | [DOI](https://doi.org/10.1145/3711896.3737247) |
| 2025 | Predicting Calibrated Conversion Rate of Online Advertising Using a Multi-task Mixture-of- | China Academy of Safety Sciences and Technology, China Unive | DEFER, DEFUSE, ES-DFM | 지연 방법 — 멀티태스크 | [DOI](https://doi.org/10.1007/978-981-96-1024-2_14) |
| 2025 | See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Predi | Alibaba Group (China) | DEFER, DEFUSE, ES-DFM | 응용 — 다중 어트리뷰션 (MAL) | [DOI](https://doi.org/10.1145/3746252.3761580) |
| 2025 | Towards Unbiased and Real-Time Staytime Prediction for Live Streaming Recommendation | Renmin University of China | DEFER, DEFUSE, ES-DFM | 응용 — 체류시간 지연 라벨 | [DOI](https://doi.org/10.1145/3746252.3761570) |
| 2026 | Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce R | Alibaba Group (China) | DEFUSE | 응용 — 할인 인지 CVR | [arXiv 2607.12578](https://arxiv.org/abs/2607.12578) |
| 2026 | Deep Learning to Rank in Industrial Search Engines, Recommender Systems, and Online Advert | Tsinghua University, Wuhan University | DEFER, DEFUSE | 서베이 | [DOI](https://doi.org/10.1145/3797895) |
| 2026 | Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, I | Alibaba Group (China), Xiamen University | DEFER, DEFUSE, ES-DFM | 벤치마크+방법 (READER) | [arXiv 2601.20307](https://arxiv.org/abs/2601.20307) |
| 2026 | Discovering and Alleviating Data Leakage in Staytime Prediction for Live Streaming Recomme | Chinese University of Hong Kong, Renmin University of China | DEFER, ES-DFM | 응용 — 체류시간 | [DOI](https://doi.org/10.1145/3770855.3818187) |
| 2026 | Fast yet Accurate Learning: A Novel Joint Data Stream and Model Framework for Staytime Pre | — | DEFER, DEFUSE, ES-DFM | 지연 방법 — 스트리밍 | [DOI](https://doi.org/10.1145/3770855.3818405) |
| 2026 | Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Pr | Institute of Computing Technology | DEFER, DEFUSE, ES-DFM | 지연 방법 (TRACE) | [arXiv 2604.23197](https://arxiv.org/abs/2604.23197) |
| 2026 | Large-Scale Online Learning for Generative List Recommendation in E-commerce: An Environme | Alibaba Group (China), Renmin University of China | DEFER, ES-DFM | 타 주제 — 온라인 학습 | [DOI](https://doi.org/10.1145/3805712.3809577) |
| 2026 | MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Me | Alibaba Group (China), Nanjing University of Science and Tec | DEFER, DEFUSE, ES-DFM | 벤치마크 (MAC) | [arXiv 2603.02184](https://arxiv.org/abs/2603.02184) |
| 2026 | Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Ins | Alibaba Group (China), Alibaba Group (United States), Xiamen | DEFER, DEFUSE, ES-DFM | 벤치마크+방법 (TESLA) — ES-DFM 가중치를 그대로 사용 | [arXiv 2601.19965](https://arxiv.org/abs/2601.19965) |
| 2026 | TemporalExpertNet: Cross-Temporal Knowledge Reuse for Promotion-Aware CVR Prediction | Fudan University, Kuaishou (China), Tianjin University | DEFER, DEFUSE | 응용 — 프로모션 CVR | [DOI](https://doi.org/10.1145/3773966.3777956) |
| 2022 | Generalized Delayed Feedback Model with Post-Click Information in Recommender Systems | Nanjing University | ES-DFM, DEFER | 지연 방법 (GDFM) — 셋을 베이스라인으로 비교. **OpenAlex 미반영, 본문 확인** | [arXiv 2206.00407](https://arxiv.org/abs/2206.00407) |
| 2026 | TWICE: Two Clocks for Delayed Feedback CVR (Kuaishou) | Kuaishou | ES-DFM, DEFER, DEFUSE | 지연 방법 — Kwai 전 트래픽 배포, 셋을 베이스라인으로 비교. **OpenAlex 미반영, 본문 확인** | [arXiv 2607.25404](https://arxiv.org/abs/2607.25404) |

