# 문제



**새로 만들거나 설정을 바꾼 광고그룹은 pCVR이 안정될 때까지 며칠 걸리고, 그 사이 과대·과소예측이 발생한다.** 그 시간을 측정해서 줄인다.

### 구조

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
> 각 원인(C1~C4)에 대해 **문제점을 지적한 논문**과 **해결 방안을 제시한 논문**을 대표적인 것만 골라 정리한다. 인용은 원문 그대로 두고 한 줄 해석을 붙였다.

### C1 — 라벨 성숙

> **문제** : 전환은 클릭하자마자 오지 않는다. 며칠 뒤에 오기도 한다(최대 7일). 그래서 어제 클릭에 붙은 "전환 안 함"은 진짜 안 샀다는 뜻이 아니라 **아직 모른다**는 뜻이다. 배치 DFM은 이걸 알고 있어서 최근 클릭의 "안 샀다" 신호를 거의 안 믿는다 — 얼마나 믿을지가 $F(d) = 1 - e^{-\lambda d}$ 이고, $\lambda = 0.7/\text{일}$ 이면 2시간 전 클릭은 5.6%만 반영한다. 신규 그룹은 가진 데이터가 전부 "최근"이라 이 할인을 그대로 다 맞는다.  
온라인 모델(`fnw`)은 정반대로 일단 전부 "안 샀다"로 즉시 학습하기 때문에 첫 며칠은 무조건 과소예측이 난다. 전환이 실제로 며칠 걸려 오는 것 자체는 앞당길 수 없으니 **없앨 수 없는 최소 지연**이다. 다만 아래 논문들은 두 가지를 더 말한다 — (a) 초반에 보이는 라벨은 그냥 적기만 한 게 아니라 **빨리 사는 사람 쪽으로 기울어 있고**(C1-b), (b) 장바구니처럼 빨리 오는 신호를 쓰면 일부는 당겨올 수 있다.


#### 문제점 지적

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

### C2 — 표본 희석 · 낡은 prior

> **문제** : 신규 그룹의 임베딩은 미학습 상태(콜드스타트)라 예측은 사실상 **과거 30일의 다른 그룹들로 학습된 공유 파라미터가 내놓는 "평균적인 그룹"의 값**이 되고, 그래서 이 그룹의 실제 CVR과 어긋난다. 둘로 갈린다 — **C2-①** 임베딩이 아직 안 배워짐(30일 창과 무관, 7일이어도 같음) / **C2-②** 그동안 받는 기본값(공유 파라미터)이 지난 30일 평균이라 **오늘과 어긋남**(창이 길수록 커짐). 학습 기간을 30→7일로 줄이면 비중은 오르지만 라벨 미성숙 샘플 비율도 올라 C1이 악화되므로, 위키 처방은 "기간을 자르지 말고 최근 데이터에 가중치".


#### 문제점 지적

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

### C1, C2 소결

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

> **한계**  
① 조회 시각 $\delta_j$ 만큼의 지연은 남는다지연을 없애는 게 아니라 며칠에서 분 단위로 옮기는 것.  
② 행동이 무정보면 성능이 안 떨어질 뿐 이득도 없다 — **우리 로그에서 장바구니·구매 간 조건부 엔트로피를 먼저 재야 판단 가능.** 

#### ES-DFM — 대기 창을 설계 변수로

> **ES-DFM** (Yang et al., AAAI 2021) — https://arxiv.org/abs/2012.03245

**로직** — 클릭이 오면 정해둔 시간 $e$ 만큼 기다렸다가, 그 안에 전환이 왔으면 양성으로, 안 왔으면 음성으로 라벨을 붙여 흘려보낸다. 창이 닫힌 뒤에 전환이 도착하면 양성 복사본을 한 번 더 넣는다. 이렇게 만들어진 왜곡된 분포를 보조 모델 둘로 되돌린다.

- $p_{dp}$ — 창이 닫힌 뒤에 전환할 확률
- $p_{rn}$ — 관측된 음성이 진짜 음성일 확률

가중치는 양성에 $1 + p_{dp}$, 음성에 $(1 + p_{dp})\,p_{rn}$. 둘 다 확률이라 $[0,1]$ 안에서만 만들어진다. 대기 시간 $e$ 를 **설계 변수**로 올린 것이 이 논문의 기여다 — 기다릴수록 라벨은 정확해지고 데이터는 낡는다는 교환을 손잡이로 만들었다.

**FNW는 $e = 0$ 인 특수해다.** 창이 0이면 모든 전환이 창 밖 전환이라 $p_{dp} = p$, 관측 음성은 전체 클릭이라 $p_{rn} = 1 - p$. 대입하면 양성 $1+p$, 음성 $(1+p)(1-p)$ 로 FNW 가중치와 정확히 같아진다. 현행 `_fnw`가 이 지점이다.

**C1에 대한 대응** — 현행 온라인 3종은 $e = 0$ 이라 모든 클릭이 일단 오답으로 시작한다. 창을 1시간만 둬도 그 안에 오는 전환은 처음부터 양성으로 들어가 가짜 미전환 비율이 크게 떨어진다. 가중치가 보조 모델의 확률 두 개로만 만들어져, `_fnw`처럼 모델 자기 예측 $\hat p$ 를 되먹이는 구조도 사라진다. **배치가 아니라 온라인 라인을 고치는 카드**라 GDFM·ULC와 겹치지 않는다.

**C2에 대한 대응** — 없음. 중요도 가중은 이미 관측된 샘플 사이에서 무게를 재분배할 뿐이라, 재분배할 관측 양성이 없는 신규 그룹에는 닿지 않는다. 오히려 창만큼 데이터가 늦어져 약하게 역행한다.

#### ULC — 가중치 대신 라벨을 고친다

> **ULC** (Wang et al., KDD 2023) — https://arxiv.org/abs/2307.12756

미전환 샘플마다 **전환 확률 라벨** $w_i = P(c_i = 1 \mid x_i, e_i, v_i = 0)$ 을 붙여 손실을 세 항으로 구성한다. 관측 시점 $T$ 이전에 전환이 확인된 샘플 항, 그리고 관측상 미전환인 샘플을 $w_i$ 만큼 전환자로 센 항과 $1 - w_i$ 만큼 진짜 미전환자로 센 항이다. 뒤의 두 항은 **같은 샘플을 확률로 쪼갠 것**이다. $w_i$ 가 정확하면 진짜 라벨로 학습한 손실과 기댓값이 같다(Theorem 1).

$$\mathcal{L}_{LC} = \frac{1}{|D|}\sum_i \big[\, v_i \log f + w_i(1-v_i)\log f + (1-w_i)(1-v_i)\log(1-f) \,\big]$$

$w$ 를 학습할 데이터는 원본에 없으므로 **counterfactual labeling**으로 만든다. 실제 데이터 수집 마감보다 $\tau$ 만큼 앞선 가상의 마감을 두고, 그 시점에 미전환이던 클릭 중 두 마감 사이에 전환한 것을 $w = 1$, 나머지를 $w = 0$ 으로 라벨링한다. 보조 모델은 이 데이터로 평범한 이진 분류 학습을 하고, 추론 시 연속 확률 $w$ 를 내놓는다. 보조 모델의 임베딩은 CVR 모델에서 복사해 초기화하며, 논문 실험에서 **교대 학습 1회로 이득의 대부분**이 나왔다.

**FNW와의 관계** — 둘 다 "아직 전환이 안 온 클릭은 완전한 0도 1도 아니다"에서 출발하는 soft label 계열이다. 차이는 둘이다. ① FNW는 음성 줄과 미래의 양성 복사본 **두 줄에 나눠** 적고, ULC는 한 줄에 $w : 1-w$ 로 적는다(샘플당 질량 합 = 1). ② 숫자의 출처가 다르다 — FNW는 모델 자기 예측 $\hat p = P(c=1 \mid x)$ 로 **경과 시간을 보지 않고**, ULC는 별도 모델이 $e$ 와 "지금까지 미전환"을 조건으로 받아 계산한다.

**C1에 대한 대응** — 배치 DFM의 감쇠는 "아직 모름"의 **영향력을 깎는** 방식이라 기울기까지 함께 사라진다. ULC는 영향력을 깎지 않고 **값을 고친다.** 이상적 $w$ 아래 편향 없음이 보장된다.

**C2에 대한 대응** — 재가중은 가짜 미전환과 비슷한 **관측 양성의 무게를 올려** 간접 보충하는데, 신선한 가짜 미전환에는 비슷한 관측 양성이 없을 수 있다(위 C2 문제점 지적의 인용). 라벨 교정은 그 그룹 **자신의** 미전환 샘플을 직접 고치므로 과거 양성에 기대지 않는다.

**성능** — Criteo 백본 4종 평균으로 Vanilla와 Oracle 사이 간격을 AUC 77.6%, log loss 83.1% 메웠다(MLP 기준 RI-AUC: FSIW 0.545 → ULC 0.837). nnDF는 전역 의존성 때문에 미니배치 최적화가 불가능해 Vanilla보다 크게 나빴다(RI-AUC −5.8) — **우리 학습 구조에서는 사실상 후보가 아니다.**

> **한계**  
> ① 보조 모델 자신도 지연 피드백을 겪는다 — $\tau$ 보다 늦게 전환하는 샘플은 음성으로 잘못 학습되며, 논문은 이를 향후 과제로 남긴다.  
> ② $\tau$ 에 최적점이 있다(Criteo 약 1주). 짧으면 오라벨이 늘고, 길면 보조 모델 학습 데이터가 낡아 **신선한** 가짜 미전환을 못 고친다.  
> ③ 실험이 전부 오프라인 배치 설정이다. **GDFM은 스트리밍, ULC는 배치** — 우리 쪽에서도 ULC는 배치 DFM 라인, GDFM은 온라인 라인에 대응한다.  
> ④ 공동 학습은 오히려 나빠졌고(초반의 부정확한 보조 모델이 CVR 모델을 오도), CVR 예측값으로 잠재 양성을 찾는 전략도 효과가 없었다(진짜 음성 대비 1:50, 노이즈만 유입).

#### 역할 분담

| | C1 | C2 |
|---|---|---|
| ES-DFM | 온라인 라인의 직접 해결책 | 해당 없음, 약하게 역행 |
| GDFM | 대기 시간을 분 단위로 단축 | 신선도 가중으로 정면 대응 |
| ULC | 미전환에 확률값 부여 | 신규 그룹 자기 데이터로 보정 |

### C3 — 파라미터 갱신량

> **문제** : 데이터도 있고 라벨도 왔는데 **한 번 학습할 때 파라미터가 움직이는 폭이 작으면** 여전히 오래 걸린다. 온라인 학습률은 APP·PUR 1e-4, MEM·PF 1e-3이고 조정 스케줄이 없다. 배치는 ClippyAdagrad lr 0.01에 Adagrad 누적기가 run마다 리셋된다. **재학습 주기(배치 4시간·온라인 30분)는 원인이 아니다** — 자주 학습하는 것과 빨리 따라잡는 것은 다르고, 남는 변수는 회당 갱신량이다.

해당 논문 없음. 지연 피드백 문헌은 손실 함수와 라벨 설계를 다루고 학습률·옵티마이저는 다루지 않는다. **기존 파라미터와 lr 값을 확인한 뒤 미세조정으로 판정한다.**

### C4 — 초기 표현 (제안)

> **문제** : 새 그룹이 생기면 group_id 임베딩이 $N(0,\ 10^{-5})$ 에서 시작한다(`models/model/feature.py`의 `OneHotFeature`, `RN_STDDEV = 0.00001`). 그 그룹은 어느 계정·캠페인·목적인지 다 알려져 있는데 그 정보를 하나도 안 쓰고 **빈 종이로 출발**한다. C1~C3가 전부 완벽해도 **첫 예측은 출발점이 결정**하며, $N$ 은 그룹 생성 순간부터 재므로 첫 몇 시간은 이 문제다.

**의미 있는 값으로 초기화하면 초기 pCVR 안정에 도움이 될 수 있다.** 초기값 후보는 아래 순서로 본다.

| | 초기값 | 평가 |
|---|---|---|
| 1 | 전역 group_id 임베딩 평균 | 사실상 무효 — 지금도 예측이 "평균적인 그룹" 값이라 달라지는 게 없다 |
| 2 | **같은 계정·캠페인 형제 그룹 임베딩의 평균** | 싸고 정보가 있다. 형제가 없으면 계정 → 목적 순으로 폴백. **첫 실험감** |
| 3 | 속성(계정·캠페인·목적·소재)을 받아 임베딩을 생성하는 모델 | MetaEmb 방식. 가장 강하지만 학습 파이프라인이 하나 늘어난다 |

**구현 시 주의 두 가지.** ① 초기화 지점이 한 곳이 아니다 — 모델 생성 시점의 `torch.nn.init.normal_` 외에 `utils_online/export.py`의 슬롯 재사용 경로(`reset not used embedding`)와 `cmd/reset_embed.py`도 같이 고쳐야 한다. 안 그러면 은퇴한 슬롯을 물려받은 새 그룹은 여전히 $10^{-5}$ 로 시작한다. ② 옵티마이저 모멘트를 0으로 민 상태에서 출발값만 커지면 초반 갱신 동역학이 바뀐다.

**측정**: 초기화 방식별로 그룹 생성 후 경과 시간별 캘리브레이션 오차 곡선을 겹쳐 그린다. 첫 몇 시간 구간의 차이가 C4의 크기다.

> **참고**: 콜드스타트 초기화를 다룬 논문은 MetaEmb(SIGIR 2019, CTR), **지연 피드백 문헌 밖**이다.

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
- **효과**: $w$ 가 정확하면 oracle 손실의 불편 추정(Theorem 1). 신규 그룹에도 잘 먹힘(위 C1·C2 절). 지금 delay 헤드 자리에 헤드 하나 추가하면 임베딩 공유는 자동. LC 손실은 세 항의 계수가 모두 음이 아니므로 nnDF식 `max(0,·)` 방어는 불필요

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
