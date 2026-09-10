# 문제

## 문제 정의

> 출처: `delayed_feedback_task.md` §1, 원 위키 [2026.09.01 전환 지연 모델링 — 현행 DFM/DFOM 정리와 개선 과제 검토](https://wiki.daumkakao.com/spaces/adrec/pages/2233805119)

### 한 문장

**새로 만들거나 설정을 바꾼 광고그룹은 pCVR이 안정될 때까지 며칠 걸리고, 그 사이 과대·과소예측이 발생한다.** 그 시간을 측정해서 줄인다.

### 구조 — 줄일 것 하나, 지킬 것 둘

| | 내용 |
|---|---|
| **목적** | 신규·설정변경 광고그룹의 캘리브레이션 도달 시간 $N$ 단축 |
| **제약 ①** | 성숙 구간 **캘리브레이션** 비열위 |
| **제약 ②** | 성숙 구간 **RIG** 비열위 |

"신규 그룹을 빨리 맞히게 만들되, 이미 잘 맞히던 그룹을 망가뜨리지 말 것." 무엇을 내주면 안 되는지가 미리 정해져 있어야 실험 결과를 판정할 수 있다. 가드레일을 AUC가 아니라 RIG로 두는 이유는 이 과제가 **캘리브레이션 문제**이기 때문이다 — AUC는 순위만 보므로 pCVR을 전부 2배로 올려도 변하지 않는다.

### $N$ 의 정의

> 그룹 생성(또는 설정 변경) 후 **캘리브레이션 오차가 ±X% 안에 Y일 연속 머무는 최초 시점**

- "연속 Y일"이 핵심 — 우연히 하루 맞은 것을 도달로 세지 않는다
- 누적 노출·클릭은 트래픽 규모가 다른 그룹을 비교하기 위한 보조 축

### 측정 방법

- **전체 트래픽 평균이 아니라 "그룹 생성 후 경과일"로 묶어서** 캘리브레이션 곡선을 그린다. 전체 평균은 성숙 그룹이 압도적으로 많아 신규 그룹의 어긋남이 씻겨 나간다
- 적응 속도는 **학습이 끝나는 속도가 아니라 예측이 맞아지기까지의 시간**이다 — 학습은 30분마다 돌아도 예측은 며칠 안 맞을 수 있다
- 목적별로 따로 잰다 — PURCHASE는 배치 DFM, APP/PF/MEM은 온라인 FNW로 메커니즘이 다르다
- age 버킷의 actual(실제 전환)은 **7일 성숙 후** 집계한다 — 미성숙 집계는 분모가 작아 가짜 과대예측 패턴을 만든다

### 아직 안 된 것

1. **$N$ 을 측정한 적이 없다** — 원인 후보를 논하기 전에 곡선부터 그려야 한다
2. **$N$ 일 동안의 손실액을 모른다** — 이 값 없이는 과제 우선순위를 정당화할 수 없다

### 원인 후보

위키의 C1(라벨 성숙)·C2(표본 희석)·C3(파라미터 갱신량)에 C4(초기 표현)를 제안으로 더한 네 축. 각 원인의 정의와 논문 근거는 아래 절.

## 예상 원인별 논문 근거
> 각 원인(C1~C4)에 대해 **문제점을 지적한 논문**과 **해결 방안을 제시한 논문**을 대표적인 것만 골라 정리한다. 순서는 Tier(0→3), 같은 Tier면 인용수. 인용은 원문 그대로 두고 한 줄 해석을 붙였다. Tier 분류와 계보는 `delayed_feedback.md`, 원인 정의는 `delayed_feedback_task.md` §2 참고.

### C1 — 라벨 성숙

> **문제** (`delayed_feedback_task.md` §2.2): 전환은 클릭 후 최대 7일까지 걸린다. 어제 클릭은 "전환 안 함"이 아니라 **"아직 모름"** 이다. DFM은 최근 클릭의 미전환 신호를 $F(d) = 1 - e^{-\lambda d}$ 만큼만 반영하는데($\lambda = 0.7/\text{일}$ 기준 2시간 전 클릭은 5.6%), 신규 그룹은 데이터가 전부 "최근"이라 이 감쇠를 정면으로 맞는다. 온라인 모델(`fnw`)은 반대로 미전환을 전 가중치로 즉시 학습해 첫 며칠은 구조적으로 과소예측한다. 전환이 실제로 7일 걸려 오는 것은 앞당길 수 없으므로 **제거 불가능한 최소 지연** — 단, 아래 논문들은 (a) 초반 라벨이 적을 뿐 아니라 **편향돼** 있고(C1-b), (b) 빠른 대리 신호로 일부는 줄일 수 있음을 보인다.


#### 문제점 지적

- **FSIW** (Yasui et al., WWW 2020, Tier 1) — https://arxiv.org/abs/2002.02068
  지연 피드백을 "학습 데이터와 서빙 환경의 조건부 라벨 분포 불일치"로 처음 정식화하고, 그 결과가 **항상 과소예측**임을 증명했다.
  > "some positive instances at the training period are labeled as negative because some conversions have not yet occurred when training data are gathered. As a result, the conditional label distributions differ between the training data and the production environment."
  > "a CVR predictor would be prone to downward bias under the feedback shift because … $P(Y=1 \mid X=x) \le P(C=1 \mid X=x)$."

- **Ktena et al.** (RecSys 2019, Tier 1) — https://arxiv.org/abs/1907.06558
  연속 학습에서 "기다리는 시간 vs fake negative 비율"이 트레이드오프임을 지적. 현행 온라인 DFOM(`fnw`)의 원 논문.
  > "fresh data may not have complete label information at the time they are ingested by the training algorithm. Naive strategies which consider any data point a negative example until a positive label becomes available tend to underestimate CTR"
  > "It is also unclear what the ideal window length would be, in order to find a trade-off between the delay in model training and the fake negative (FN) rate"

- **DLA-DF** (Saito, Morishita & Yasui, SIGIR 2020, Tier 2) — https://arxiv.org/abs/1910.01847
  C1의 하위 문제: 초반에 관측되는 positive는 **빠른 전환자에 쏠려 있다**(MNAR). 라벨이 "적은" 게 아니라 "편향된" 것.
  > "decisive users are much more likely to convert immediately after a click than indecisive users. Therefore, the probabilities of conversions being observed correctly are not uniform among samples. … the MNAR mechanism can lead to sub-optimal and biased estimations"

- **DEFUSE** (Chen et al., WWW 2022, Tier 1) — https://arxiv.org/abs/2202.06472
  기존 재가중 방법들의 공통 결함: 관측 negative를 일괄 negative로 취급.
  > "observed negatives may potentially be fake negatives, and these methods falsely treat them as real negatives, leading to sub-optimal performance."

#### 해결 방안 제시

- **DFM** (Chapelle, KDD 2014, Tier 0) — https://dl.acm.org/doi/10.1145/2623330.2623634
  미전환 샘플을 버리지 않고 **"아직 안 왔을 확률"을 생존함수로 남겨 우도에 반영**. 현행 배치 DFM(`mtldfm_v2`)이 이 방식.
  $$\Pr(Y=0 \mid x, e) = 1 - p(x) + p(x)\,e^{-\lambda(x)\,e}$$

- **FNW / FNC** (Ktena et al., RecSys 2019, Tier 1)
  기다리지 않고 negative로 넣은 뒤 전환 시 positive 복사본을 추가하고, 그 왜곡을 가중치(FNW) 또는 추론 시 보정(FNC)으로 되돌린다. 현행 온라인 3종(`fnw`)이 이 방식.
  > "samples are labeled as negatives and ingested to the training pipeline, and then duplicated with a positive label as soon as a user engagement takes place."
  가중치: positive $(1+\hat p)$, negative $(1-\hat p)(1+\hat p)$ / FNC: $p = q/(1-q)$

- **ES-DFM** (Yang et al., AAAI 2021, Tier 1) — https://arxiv.org/abs/2012.03245
  대기 창을 **설계 변수**로 두고, 두 보조 확률($p_{dp}$: 나중에 뒤집힐 확률, $p_{rn}$: 진짜 negative일 확률)로 가중치를 만든다. FNW는 창=0인 특수해.
  > "there is a trade-off between waiting for more accurate labels and utilizing fresh data, which is not considered in existing works."
  가중치: positive $1+p_{dp}$, negative $(1+p_{dp})\,p_{rn}$

- **ULC** (Wang et al., KDD 2023, Tier 2) — https://arxiv.org/abs/2307.12756
  가중치 대신 **라벨 자체를 교정** — 관측 negative마다 "나중에 전환할 확률 $w$"를 붙여 학습.
  $$\mathcal L_{LC} = v\log f + w(1-v)\log f + (1-w)(1-v)\log(1-f)$$
  > "Theorem 1. If an ideal label correction model is satisfied, i.e., $w_i = P(c_i=1 \mid x_i, e_i, v_i=0)$, then the LC loss is unbiased to the oracle loss."

### C2 — 표본 희석 · 낡은 prior

> **문제** (`delayed_feedback_task.md` §2.3): 신규 그룹의 임베딩은 미학습 상태(콜드스타트)라 예측은 사실상 **과거 30일의 다른 그룹들로 학습된 공유 파라미터가 내놓는 "평균적인 그룹"의 값**이 되고, 그래서 이 그룹의 실제 CVR과 어긋난다. 둘로 갈린다 — **C2-①** 임베딩이 아직 안 배워짐(30일 창과 무관, 7일이어도 같음) / **C2-②** 그동안 받는 기본값(공유 파라미터)이 지난 30일 평균이라 **오늘과 어긋남**(창이 길수록 커짐). 학습 기간을 30→7일로 줄이면 비중은 오르지만 라벨 미성숙 샘플 비율도 올라 C1이 악화되므로, 위키 처방은 "기간을 자르지 말고 최근 데이터에 가중치".


#### 문제점 지적

- **GDFM** (Yang & Zhan, NeurIPS 2022, Tier 2) — https://arxiv.org/abs/2206.00407
  C2를 **두 개로 쪼개 정식화**한 유일한 논문. sampling gap(표본 부족 = C2-①)과 temporal gap(낡은 분포 = C2-②).
  > "(i) Estimating conversion rates via post-click actions requires more samples than using conversion labels directly, which highlights the importance of sample complexity. (ii) The post-click actions bring information of past distributions, which incurs a temporal gap."
  > "Even if we have unlimited samples from $p_{t-\delta}$ we are only able to recover $p_{t-\delta}(y \mid x)$ instead of $p_t(y \mid x)$."
  → 표본이 무한해도 **과거 분포**밖에 못 배운다. 신규 그룹이 받는 "평균값"이 낡은 이유.

- **nnDF** (Kato & Yasui, KDD 2022, Tier 2) — https://arxiv.org/abs/2009.13092
  모든 지연 보정이 깔고 있는 **정상성 가정**을 명시하고, **신규 캠페인이 그 가정을 깬다**고 직접 썼다.
  > "Assumption 2 (Stationarity Assumption). … $p(Y_i(E_i^t) \mid X_i = X, E_i^t = s) = p(Y_j(E_j^{t'}) \mid X_j = X, E_j^{t'} = s)$"
  > "This often happens in advertising platforms, such as when a new campaign is launched. Since there are fewer data of the new campaigns, the test data has some shift from the training data, so the stationarity assumption does not hold in this setting."
  → C1 보정이 C2 상황에서 틀린다. Criteo 실험에서 새 캠페인 비율에 따라 날마다 최고 방법이 바뀜.

- **ULC** (KDD 2023, Tier 2)
  재가중 계열은 **이미 있는 positive의 무게를 재분배**할 뿐이라, 과거와 다른 신규 그룹의 fake negative를 표현할 수 없다.
  > "[This problem] is worse when the data distribution has changed recently. As the information about the false negative samples may differ from the past observed positive samples, only using the observed positive samples cannot complement the correct information about the fresh false negative samples."

#### 해결 방안 제시

- **GDFM** (NeurIPS 2022, Tier 2)
  temporal gap을 **빠른 대리 신호**(장바구니 등 post-click 행동)로 줄이고, 신선도 가중 $w_{time} = e^{-\beta\delta}$·정보량 가중 $w_{info} = e^{-\alpha H(y\mid a)}$ 로 sampling gap에 대응.

- **ULC** (KDD 2023, Tier 2)
  라벨 교정은 신규 그룹 **자신의** fake negative를 soft positive로 바꾸므로, 과거 positive에 의존하지 않고 신규 그룹 정보를 즉시 넣는다. (C1 해결책과 동일 — C2 조건에서 재가중보다 유리한 이유가 위 인용)

- **AutoFuse** (Jin et al., ICDE 2023, 서베이 인용) — https://doi.org/10.1109/ICDE55515.2023.00264
  표본 없는 세밀 ID 대신 **상위 그룹 표현**을 쓴다 — C2-①과 C4의 교차점.
  > "Tail elements in those high-cardinality features … tend to have inadequate samples and thus fail to obtain semantically meaningful embeddings. … AutoFuse learns an ad-level representation to depict the unique individual character and a group-level representation to portray the collective information by discarding the fine-grained features."

> **주의**: 위키 C2 처방("최근 데이터에 가중치")을 그대로 제안한 논문은 없다. 가장 가까운 것은 MISS의 assembled pipeline(옛 positive를 최신 positive로 교체, AAAI 2024).

### C3 — 파라미터 갱신량

> **문제** (`delayed_feedback_task.md` §2.4): 데이터도 있고 라벨도 왔는데 **한 번 학습할 때 파라미터가 움직이는 폭이 작으면** 여전히 오래 걸린다. 온라인 학습률은 APP·PUR 1e-4, MEM·PF 1e-3이고 조정 스케줄이 없다. 배치는 ClippyAdagrad lr 0.01에 Adagrad 누적기가 run마다 리셋된다. **재학습 주기(배치 4시간·온라인 30분)는 원인이 아니다** — 자주 학습하는 것과 빨리 따라잡는 것은 다르고, 남는 변수는 회당 갱신량이다.


#### 문제점 지적

- **IF-DFM** (Ding et al., AAAI 2026, Tier 3) — https://arxiv.org/abs/2502.01669
  오프라인은 분포 변화에 못 따라가고, 온라인은 복제 샘플 때문에 정확한 라벨을 제대로 못 쓰며, 전체 재학습은 비용 때문에 불가 — **"갱신"이 병목**이라는 진단.
  > "While online methods update with new data promptly, their reliance on duplicated samples for label correction can cause confusion and limit the effective use of accurate labels."
  > "While retraining with correctly labeled data is straightforward, it is impractical in large-scale CVR settings due to high computational costs."

- **Ktena et al.** (RecSys 2019, Tier 1)
  갱신 지연 자체가 성능을 깎는다는 실무 관측.
  > "Internally, empirical results show that even a 5-minute delay to [model updates hurts serving performance]"
  (위키가 "재학습 주기는 원인이 아니다"로 정리했듯, 이는 주기가 아니라 **회당 갱신이 얼마나 반영되느냐**의 문제로 읽어야 한다.)

#### 해결 방안 제시

- **IF-DFM** (AAAI 2026, Tier 3)
  늦게 도착한 전환(라벨 반전)이 파라미터에 주는 영향을 **influence function으로 직접 계산해 갱신**. 재학습·보조 모델·샘플 복제 없음.
  > "The core idea is to leverage influence functions to estimate the impact of newly injected data and directly update model parameters without retraining. … This avoids sample duplication and eliminates the need for auxiliary models."
  $$\Delta\theta \approx -\tfrac{1}{n}H^{-1}\Big[\textstyle\sum_{j\in J}\big(\nabla L(z_j^\delta)-\nabla L(z_j)\big)+\sum_{k\in K}\nabla L(z_k)\Big]$$

- **MetaEmb** (Pan et al., SIGIR 2019, 서베이 인용) — https://arxiv.org/abs/1904.11547
  좋은 초기값이 **warm-up 수렴을 가속**한다 — C4가 C3를 돕는 경로.
  > "the generated embedding can speed up the model fitting during the warm-up phase when a few labeled examples are available, compared to the existing initialization methods."

- **서베이** (Xue, Yang & Zhai, 2025, Tier 3) §6.3.3 — https://arxiv.org/abs/2512.01171
  > "meta-learning enables models to extract valuable knowledge from a small number of samples and quickly adapt to various tasks that have not been encountered before … dynamically adapting modeling parameters through meta-optimization that leverages delayed conversion data across multiple time windows."

> **주의**: 위키 C3 처방("신규 그룹 임베딩에만 큰 학습률")을 다룬 논문은 목록에 없다. 이 항목은 논문이 아니라 **사내 실험(lr·Clippy `lambda_abs`·`forget`)** 으로 판정해야 한다.

### C4 — 초기 표현 (제안)

> **문제** (`delayed_feedback_task.md` §2.5): 새 그룹이 생기면 모델은 그 임베딩을 $N(0,\ 10^{-5})$, 사실상 **0에서** 시작한다. 그런데 그 그룹은 아무것도 모르는 상태가 아니다 — 어느 계정 밑인지, 무슨 캠페인·목적인지, 형제 그룹들이 어떻게 전환됐는지 다 알면서 그 정보를 하나도 안 쓰고 **"빈 종이"로 출발**한다. C1~C3가 전부 완벽해도(채점 즉시·기본값 정확·갱신 즉시) **첫 예측은 출발점이 결정**하며, $N$ 은 그룹 생성 순간부터 재므로 첫 몇 시간은 순수하게 이 문제다. 위키가 "아직 축으로 세우지 않은 후보 — 임베딩 초기화"로 적어둔 것을 서베이 근거로 승격한 제안이다.


#### 문제점 지적

- **MetaEmb** (SIGIR 2019, 서베이 인용)
  ID 임베딩은 데이터를 많이 요구해서 **새 광고에서는 작동하지 않는다** — 콜드스타트를 임베딩 문제로 명명.
  > "such learning techniques are data demanding and work poorly on new ads with little logging data, which is known as the cold-start problem."

- **AutoFuse** (ICDE 2023, 서베이 인용)
  피처마다 콜드스타트를 다르게 겪는다 — 세밀 ID(광고·그룹)는 의미 없는 임베딩이 되고, 그게 **다른 피처와의 교차까지 오염**시킨다.
  > "different features suffer differently from cold-start issues. Tail elements in those high-cardinality features, which we denote as fine-grained features, tend to have inadequate samples and thus fail to obtain semantically meaningful embeddings. Interacting with those features leads astray and impairs the accuracy of new ads in a cold-start scenario."

- **서베이** (2025, Tier 3)
  CVR 문헌이 **new ads / old ads, warm / cold**로 성능표를 갈라서 잰다(Moment·Weapp 데이터셋) — 신규 광고가 별도 평가 축임을 보여준다.

#### 해결 방안 제시

- **MetaEmb** (SIGIR 2019)
  새 ID의 **초기 임베딩을 생성하는 생성기**를 메타러닝으로 학습. 콘텐츠·속성 → 임베딩.
  > "learns to generate desirable initial embeddings for new ad IDs. The proposed method trains an embedding generator for new ad IDs by making use of previously learned ads through gradient-based meta-learning. … When a new ad comes, the trained generator initializes the embedding of its ID by feeding its contents and attributes."

- **AutoFuse** (ICDE 2023)
  세밀 ID를 **버린** 상위 그룹 표현을 따로 배워 광고 표현과 적응적으로 융합 — 새 광고는 그룹 표현이 예측을 담당. 산업 배포.
  > "The final robust and general ad representation is obtained by integrating these two level representations adaptively. Such a combination encompasses a wider amount of information, and thereby mitigates the cold-start issue."

- **DCBT** (Yang et al., SIGIR 2023, 서베이 인용) — https://doi.org/10.1145/3539618.3591856
  초기값은 그대로 두고 **같은 배치의 warm 샘플에서 cold 샘플 표현을 보강** — 같은 문제, 다른 레버.
  > (서베이 §4.3.4) "introduced a Transformer module to enhance cold sample representations by extracting related information from warm samples in the same batch"

- **MVTA** (Yao et al., IEEE TBD 2023, 서베이 인용) — https://doi.org/10.1109/TBDATA.2022.3162150
  새 캠페인을 소재·전환 규칙·타겟팅으로 임베딩해 **이웃 캠페인의 CVR로 예측**. 캠페인 단위 forecasting.
  > "learning an unsupervised and composite campaign embedding to capture multi-view semantic relationships on campaign information, and consequently forecasting the cold-start campaigns using the nearest neighbor campaigns."

> **주의**: C4 해결책은 전부 **지연 피드백 문헌 밖**(콜드스타트 문헌)에서 온다. MetaEmb는 CTR 논문이며, 서베이가 CVR 데이터셋 baseline으로 비교했다.

# 모델 성능 향상 제안

> 출처: `suggestion.md`. 신규 그룹 $N$ 단축과 무관하게 **현행 모델의 일반 성능**을 올릴 수 있는 논문 아이디어 중, **Tier 1 이상 또는 빅테크 배포 논문**만 골랐다. 같은 레버를 당기는 것끼리 A~F로 묶었다.

## A. 지연 분포(λ) 헤드 — 배치 DFM

### A1. 지수분포 → 버킷 CDF
> NoDeF (Yoshikawa & Imai, 2018, Tier 0) — https://arxiv.org/abs/1802.00255 · TWICE (Kuaishou, 2026, 전 트래픽 배포) — https://arxiv.org/abs/2607.25404

- **지금**: `MTLCrossv2dfm`의 Delay 헤드가 λ 하나 → 지수분포. "클릭 직후에 살 확률과 3일 뒤에 살 확률이 같다"는 가정. 24시간 주기·고가 상품의 숙고 구간을 못 담음
- **바꿀 것**: "1일차·2일차·…·7일차에 살 확률" $K$개를 softmax로 → 누적합이 단조 CDF. `timeunit=86400`, `max_delay=7`과 맞물림. TWICE의 delay head가 정확히 이 구조
- **효과**: 미전환 샘플을 얼마나 믿을지가 정확해짐. loss와 헤드 차원만 바꾸고 파이프라인은 안 건드림

### A2. λ를 가짜 미전환에서 보호
> TWICE (Kuaishou) — A/B에서 기대 매출 +2.486%, 전환 +2.061%

- **지금**: 미전환 항 `log(1 − p + p·e^{−λd})`가 p와 λ를 동시에 밈 → 아직 안 온 전환이 λ를 왜곡
- **바꿀 것**: 미전환 항에서 λ로 가는 기울기를 끊음(`λ.detach()`) → λ는 실제 도착한 전환에서만 배움. λ 입력은 전체 피처 대신 안정적인 몇 개(목적·캠페인·계정·지면)로
- **효과**: λ가 흔들리지 않음. 단 "더 옳다"가 아니라 "더 강건하다"는 교환 — A/B로 판정

### A3. 버려지는 λ로 임의 창 CVR 서빙
> Personalized Interpolation (Meta, CIKM 2025) — https://arxiv.org/abs/2501.14103

- **지금**: λ를 학습만 하고 추론에서 버림. 서빙은 7일 창 하나인데 vcvr은 1일, 나머지는 7일로 목적마다 창이 다름
- **바꿀 것**: `(p, λ)`를 함께 내보내고 서빙이 목적별 창 $T$ 를 넣어 계산. 지수분포 가정 아래 닫힌형 $P(\text{T일 안에 전환}\mid x) = \hat p(x)\,(1 - e^{-\hat\lambda(x)T})$ — Meta의 보간 계수와 같은 식
- **효과**: **학습 변경 0**으로 1일·3일·7일 CVR을 한 모델에서. $\hat p(1-e^{-\hat\lambda})$ 와 실측 1일 전환율을 비교하면 λ 캘리브레이션 진단도 공짜. 가장 싸다

## B. 관측 창·헤드 설계

A가 λ·검열항을 *잘 만들자*라면 B2는 *아예 안 쓰자*다. 배치 라인에서 둘을 대조군으로 세우면 어느 쪽이 맞는지 한 실험으로 갈린다.

### B1. 온라인 모델에 "잠깐 기다리기" + 지연 보조 헤드
> ES-DFM (Alibaba, AAAI 2021, Tier 1) — https://arxiv.org/abs/2012.03245

- **지금**: 클릭이 오면 0초 기다리고 바로 "전환 안 함"으로 학습. 모든 전환이 일단 오답으로 시작 (`_fnw`는 ES-DFM의 $e=0$ 특수해)
- **바꿀 것**: 1시간 정도 기다린 뒤 라벨 붙이기 + "나중에 전환할 확률" 헤드($f_{dp}, f_{rn}$). `_fnw` 가중치 `(1+p̂)`, `(1−p̂)(1+p̂)` → `(1+f_dp)`, `(1+f_dp)·f_rn`
- **효과**: 가짜 미전환이 크게 줄고 가중치가 $[0,1]$ 확률로만 만들어져 폭발하지 않음. 스트림 emit 시점 변경이 비용

### B2. 창별로 "확정된 라벨"만 써서 학습
> FTP (CAS·Tencent, SIGIR 2021) — https://arxiv.org/abs/2108.06167 · Defer 오프라인 버전 (Alibaba, ESDF §3.2) · DEFUSE (Alibaba, WWW 2022, Tier 1) — https://arxiv.org/abs/2202.06472

- **지금**: 모든 헤드가 "아직 모르는" 샘플을 안고 배우고 그걸 가중치나 λ로 사후 보정
- **바꿀 것**: 헤드를 "1시간 안에 / 1일 안에 / 7일 안에 전환"으로 나누고 **각 헤드는 그 시간이 이미 지난 샘플로만 학습** → 가짜 미전환이 아예 없음. 합칠 때는 7일 지난 데이터로 만든 완벽한 모델(prophet)에 가장 가까운 헤드를 고르는 aggregator. DEFUSE의 IP/DP 분리는 창 2개짜리 특수형 — 창 안 전환(IP)은 보정 없이 순수 BCE
- **효과**: 보정 가중치·λ·검열항 전부 불필요. Alibaba가 실제로 쓰는 구조(Defer n+1 tower). `MTLSimpledfom`·`MTLCrossv2dfm`이 이미 action_type별 헤드라 헤드를 K배로 늘리고 합산 규칙만 바꾸면 됨

### B3. 진짜 미전환도 다시 넣기
> DEFER (Alibaba, KDD 2021, Tier 1) — https://arxiv.org/abs/2104.14121

- **지금**: 전환한 샘플만 나중에 한 번 더 넣음 → 전환 잘 되는 피처 영역만 부풀려짐 ($q(x)\ne p(x)$)
- **바꿀 것**: 어트리뷰션 창(7일) 지나면 **모든 클릭**을 확정 라벨로 한 번 더 넣기
- **효과**: 데이터 왜곡 제거 + "확실히 안 산다" 정보 확보. FNW 변형이면 positive 가중치가 상수 2, FNC면 `2q`로 단순화. 데이터 2배가 비용

### B4. 라벨 소스별 헤드를 "그냥 더하기" 대신 비대칭 전이로
> MAC / MoAE (Alibaba, KDD 2026) — https://arxiv.org/abs/2603.02184

- **지금**: `OBJECTIVE_MAP["PURCHASE"] = ["bcon_Purchase", "Purchase"]` — 두 라벨 소스 헤드를 `output_mask`로 단순 합산
- **바꿀 것**: 과금 기준 라벨을 주 헤드로, 나머지는 보조로 두고 **보조 → 주 한 방향으로만** 전이. MAC의 발견: 여러 어트리비션 라벨을 함께 배우면 좋아지지만 보조가 주를 오염시키면 오히려 나빠짐
- **효과**: 헤드는 이미 분리돼 있어 합산 규칙만 바꾸면 됨. 1일 창 vs 7일 창 라벨에도 같이 적용 가능. (창별 헤드의 동적 합산은 MISS의 synthesizer — 2층 MLP + softmax — 참고)

## C. 라벨 교정

### C1. 과거 로그로 미래 라벨 만들기
> FSIW (CyberAgent, WWW 2020, Tier 1) — https://arxiv.org/abs/2002.02068

- B1·B2·C2에 필요한 "이 클릭이 창 안에 전환할까 / 나중에 뒤집힐까" 라벨을 어떻게 구하나?
- **방법**: 지금 로그의 시계를 $\tau$(예: 7일) 전으로 되감아 "그때 기준으로 창 안이었나"를 표시(counterfactual deadline). 기다릴 필요 없이 지금 데이터로 보조 헤드 학습셋을 만듦. 배치 잡 하나

### C2. 미전환에 "전환 확률" 라벨을 붙여서 학습
> ULC (Tsinghua·Huawei, KDD 2023) — https://arxiv.org/abs/2307.12756

- **지금**: DFM 검열항은 "관측된 라벨"이 나올 확률을 최대화 — 진짜(oracle) 라벨 기준이 아님
- **바꿀 것**: 보조 모델이 미전환 샘플마다 "나중에 $w$ 확률로 전환"이라고 라벨 붙임. $\mathcal L_{LC} = v\log f + w(1-v)\log f + (1-w)(1-v)\log(1-f)$. 보조 모델은 CVR 모델의 임베딩을 복사해서 시작, 둘을 번갈아 학습
- **효과**: $w$ 가 정확하면 oracle 손실의 불편 추정(Theorem 1). 신규 그룹에도 잘 먹힘(위 C1·C2 절). 지금 delay 헤드 자리에 헤드 하나 추가하면 임베딩 공유는 자동. 빼는 항이 있는 손실이라 음수 부분에 `max(0,·)` 안전장치(nnDF) 필요

## D. post-click 행동 신호

### D1. 장바구니를 "구매 예고 신호"로
> GDFM (Nanjing Univ., NeurIPS 2022) — https://arxiv.org/abs/2206.00407 · 후속 TRACE (CAS, SIGIR 2026) — https://arxiv.org/abs/2604.23197

- **지금**: 장바구니는 클릭 후 몇 분, 구매는 며칠. `OBJECTIVE_MAP`에 CART 액션이 있지만 `mtldfm_v2`는 positive가 자기 action_type 헤드만 갱신 → 장바구니가 구매 예측을 전혀 안 건드림
- **바꿀 것**: "장바구니에 담은 사람 중 몇 %가 사는지"를 배워서 장바구니가 관측되면 그 순간 구매 확률을 미리 올려놓기. 가중치 = 정보량(조건부 엔트로피) × 신선도($e^{-\beta\delta}$). TRACE는 이를 "10분 뒤 장바구니 → 1시간 뒤에도 그대로 → 1일 뒤 미구매" 같은 누적 궤적으로 확장
- **효과**: 구매 pCVR이 며칠 뒤가 아니라 몇 분 뒤에 반응. Alibaba 로그에서 구매의 50%가 직전 행동 1시간 내. 장바구니 데이터는 이미 있고 연결만 없음

## E. 구매 모델의 배수 축 (mpc)

mpc 출력 = pCVR × `conv_multiplier`. 이 배수를 위(재구매)와 아래(환불)에서 각각 고친다. 세 편 모두 Alimama 형제 논문.

### E1. 재구매 배수를 그룹 스칼라 대신 클릭별 예측으로
> READER (Alibaba, WWW 2026) — https://arxiv.org/abs/2601.20307

- **지금**: 배수는 group_id별 정적 스칼라, 오프라인 회귀. 새 그룹은 1.0
- **바꿀 것**: ① 이 클릭이 2회 이상 구매로 이어질 확률을 예측하는 router ② 단일구매/재구매 타워 2개 ③ 창이 닫히기 전 관측 누적값에서 최종값을 추정하는 label calibrator → 온라인 학습 가능
- **효과**: Taobao는 전환 클릭의 54%가 재구매 — 그룹 스칼라로는 못 잡음. 새 그룹 배수=1.0 문제도 함께 풀림. econv 파이프라인에 그대로 들어감

### E2. 환불 빼기 — ECVR
> ESDF (Alibaba, CIKM 2023, 배포 ECVR +5.21%) — https://arxiv.org/abs/2308.04768 · TESLA (Alibaba, WWW 2026) — https://arxiv.org/abs/2601.19965

- **지금**: 재구매 배수는 곱하는데 **환불은 안 뺌**
- **바꿀 것**: 전환 샘플 위에 환불률 헤드 + 환불 어트리뷰션 창 → $p_{ecvr} = p_{cvr}\times(1-p_{rfr})$. 환불 라벨은 전환 샘플에만 있어 SSB·DS가 심하니 ESMM식으로 전 클릭 공간에서 함께 학습. 온라인이면 TESLA의 stage-wise 중요도 가중
- **조건**: 환불 로그가 있어야 함. 확인 먼저

## F. 학습·서빙 기타

### F1. FNC 한번 켜보기
> Ktena et al. (Twitter, RecSys 2019, Tier 1) — https://arxiv.org/abs/1907.06558

- 코드에 이미 있는데(`forward_recalibration`) 전부 꺼져 있음. 가중치 대신 추론 때 `q/(1−q)`로 보정
- **효과**: 논문상 FNW와 성능 비슷. 가중치가 모델 자기 예측에 의존하는 구조를 없앨 수 있음. config 한 줄(`recalibration: true` + `loss_info.type: BCELoss`). **FNW와 동시에 켜면 이중 보정**이므로 택일

## 순서

| 순서 | 항목 | 이유 |
|---|---|---|
| **1** | A3 λ로 임의 창 서빙 (Meta) | **학습 0**. λ 진단도 공짜 |
| 2 | A1 + A2 버킷 CDF + λ 보호 (Kuaishou) | loss·헤드만. 파이프라인 안 건드림 |
| 3 | C1 → B2 라벨 생성 → 창별 확정 헤드 (Alibaba) | 가짜 미전환을 학습에서 제거. MTL 골격 재활용 |
| 4 | B4 비대칭 합산 (Alibaba) | 헤드 이미 분리됨. 합산 규칙만 |
| 5 | D1 장바구니 신호 (NeurIPS) | 효과 가장 클 가능성. 데이터 있음 |
| 6 | C2 라벨 교정 (Huawei) | 배치 검열항 대체 |
| 7 | E1 재구매 router (Alibaba) | econv 대체. 새 그룹 배수=1.0 해소 |
| 8 | B1 ES-DFM 대기 창 (Alibaba) | 파이프라인 변경 |
| 9 | B3 DEFER 재투입 (Alibaba) | 데이터 2배 |
| 조건부 | E2 환불 (Alibaba) | 로그 확인 후 |
| 언제든 | F1 FNC (Twitter) | config 한 줄 |

> **대조군 설계**: A(λ·검열항을 잘 만들자)와 B2(λ·검열항을 안 쓰자)는 배치 라인에서 서로의 대조군이다. 같은 실험에서 "지연 분포를 모델링하는 게 낫나, 창별 확정 라벨이 낫나"가 갈린다.
