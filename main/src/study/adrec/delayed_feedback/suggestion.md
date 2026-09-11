# C1 개선안 제안

> 범위: 신규·설정변경 광고그룹의 캘리브레이션 도달 시간 $N$ — 위키 C1(라벨 성숙)에 대한 논문 근거
> 관련 문서: `delayed_feedback_task.md` §2.2 C1

## 1. 가중치 조절은 새 그룹에 안 먹힌다 → 라벨 교정으로 (ULC)

> Wang et al., *Unbiased Delayed Feedback Label Correction for Conversion Rate Prediction*, KDD 2023 — https://arxiv.org/abs/2307.12756

지금 온라인 모델(`fnw`)이 하는 일은 **"이미 들어온 전환 샘플의 무게를 늘리고, 미전환 샘플의 무게를 줄이는 것"** 이다.

문제는 새 그룹이다. 새 그룹의 전환 샘플은 **아직 안 들어왔다.** 없는 것의 무게를 늘릴 수는 없다. 아무리 가중치를 잘 조절해도 "이 그룹은 이렇게 전환한다"는 정보 자체가 모델에 없다.

> "이 문제는 데이터 분포가 최근에 바뀌었을 때 더 심하다. 가짜 negative의 정보는 과거 관측 positive와 다를 수 있으므로, 관측 positive만 쓰면 신선한 가짜 negative의 올바른 정보를 보완할 수 없다" (ULC §2.2)

ULC의 방식은 다르다. 새 그룹의 미전환 샘플 하나하나에 **"이건 30% 확률로 나중에 전환할 것"** 처럼 확률 라벨을 붙여 학습시킨다. 전환이 실제로 오기 전에도 그 그룹의 정보가 들어간다. 라벨 교정용 학습 데이터는 counterfactual deadline(FSIW 방식)으로 만든다.

→ **결론**: 새 그룹 문제에는 가중치 조절(FSIW·ES-DFM·`fnw` 계열)보다 **라벨 교정(ULC·DEFUSE 계열)** 이 맞다. 위키 §2가 1순위로 든 "ES-DFM 재가중"은 새 그룹에 덜 먹힐 수 있다.

- 재가중 계열: ES-DFM https://arxiv.org/abs/2012.03245 · FSIW https://arxiv.org/abs/2002.02068
- 라벨 교정 계열: DEFUSE https://arxiv.org/abs/2202.06472 · ULC https://arxiv.org/abs/2307.12756

## 2. 지연 보정은 "세상이 안 바뀐다"는 전제 위에 서 있다 (nnDF)

> Kato & Yasui, *Learning Classifiers under Delayed Feedback with a Time Window Assumption*, KDD 2022 — https://arxiv.org/abs/2009.13092

모든 지연 보정 방법은 이런 가정을 깔고 있다: **"지난주에 클릭한 사람과 오늘 클릭한 사람은 같은 방식으로 전환한다."** 그래야 지난주 데이터로 오늘의 미전환을 보정할 수 있다.

> Assumption 2 (Stationarity): $p\big(Y(E) \mid X, E{=}s\big)$ 가 도착 시점 $t'$ 과 무관하게 같다

새 그룹은 이 가정을 **정의상** 깬다. 지난주엔 존재하지 않았기 때문이다.

nnDF가 실제 데이터(Criteo)로 보였다: 새 캠페인이 많이 들어온 날엔 보정 방법 A가 이기고, 적은 날엔 B가 이긴다. **어떤 보정이 좋은지가 새 캠페인 비율에 따라 뒤집힌다.**

> "Day 58 이후 추가된 캠페인의 CVR이 더 높다. 그 결과 비정상성의 강도가 날마다 달라 7일간 최고 성능 방법이 달랐다" (nnDF §6.2)

→ **결론**: "새 그룹이라 데이터가 적다(C2)"와 "라벨이 안 왔다(C1)"는 따로 놀지 않는다. C2가 클수록 C1 보정이 틀린다. 위키의 "C1×C2 교호작용도 함께 본다"는 약하고, **교호작용이 주범일 수 있다**로 올려야 한다.

## 3. 새 그룹의 첫 전환들은 "보통 유저"가 아니다 (DLA-DF)

> Saito, Morishita & Yasui, *Dual Learning Algorithm for Delayed Conversions*, SIGIR 2020 — https://arxiv.org/abs/1910.01847

전환 속도는 유저마다 다르다. **결정 빠른 유저는 바로 사고, 고민하는 유저는 며칠 뒤에 산다.**

> "결정이 빠른 유저는 우유부단한 유저보다 클릭 직후 전환할 가능성이 훨씬 높다. 따라서 전환이 올바르게 관측될 확률은 샘플마다 균일하지 않다 — MNAR(Missing-Not-At-Random)" (DLA-DF §1)

그러면 새 그룹 첫 며칠에 들어오는 전환은 전부 **결정 빠른 유저**다. 고민하는 유저의 전환은 아직 안 왔다. 모델은 이 편향된 표본을 보고 "이 그룹은 이런 유저가 전환한다"고 배운다.

DLA-DF는 관측 확률 $\theta(X, E) = P(O{=}1 \mid X, E)$ 를 propensity로 두고 CVR 모델과 교대로 학습한다.

| | MNAR 인식 여부 |
|---|---|
| 배치 DFM (`mtldfm_v2`) | $\lambda(x)$ 가 샘플별 → 원리상 인식 |
| 온라인 `fnw` | 가중치가 $X$ 무관 → **인식 못 함** |

→ **결론**: C1은 "라벨이 적다"만이 아니라 **"초반 라벨이 특정 유저 쪽으로 쏠려 있다"** 이다. 특히 온라인 모델(설치·가입·친구추가)에서 그렇다. 검증: 신규 그룹의 첫 positive들의 지연 분포가 성숙 후와 다른지 비교.

## 세 줄 요약

| 논문 | 한 줄 | 위키에 반영할 것 |
|---|---|---|
| **ULC** (KDD'23) | 없는 전환의 무게를 늘릴 순 없다 → 새 그룹엔 가중치 말고 **라벨 교정** | §2 C1 1순위 처방을 재가중 → 라벨 교정으로 |
| **nnDF** (KDD'22) | 보정은 "세상이 안 바뀐다"를 전제 → 새 그룹은 그 전제를 깨니 **C1과 C2가 얽힘** | §1.2 "교호작용도 본다" → "교호작용이 주효과일 수 있다" |
| **DLA-DF** (SIGIR'20) | 첫 전환들은 **빠른 유저만** → 초반 라벨은 적은 게 아니라 **편향됨** | C1을 C1-a(지연) / C1-b(조기 positive 편향, 온라인)로 분리 |

셋 다 새 원인(C4)은 아니고, **C1을 더 정확히 쓰고 처방을 바꾸는 근거**다.

---

# torch-dnn CVR 모델 개선 제안 — 논문 아이디어 적용

> 관련 문서: `delayed_feedback.md`(논문 계보), `torch-dnn_dfm_dfom.md`(현행 코드), `delayed_feedback_task.md`(개선 과제)
> 범위: 신규 광고그룹 $N$ 단축과 무관하게, **현행 모델의 일반 성능**을 올릴 수 있는 논문 아이디어
> 구성: Tier 1·2·3 아이디어를 **같은 레버를 당기는 것끼리 A~F 여섯 그룹**으로 묶었다. 중복은 한 항목으로 합쳤고 출처는 항목마다 표기.

## A. 지연 분포(λ) 헤드 — 배치 DFM

배치 mpc의 `MTLCrossv2dfm`은 CVR 헤드와 Delay 헤드(λ)를 갖는다. 이 그룹은 **λ 헤드를 고쳐서 검열항을 정확하게 만들고, 그 λ를 서빙까지 끌고 가는** 아이디어다.

### A1. 지수분포 → 버킷 CDF (NoDeF · TWICE)

> Yoshikawa & Imai, *A Nonparametric Delayed Feedback Model*, 2018 — https://arxiv.org/abs/1802.00255
> Li et al., *TWICE: Two-Clock, Two-Window Learning*, 2026 — https://arxiv.org/abs/2607.25404

**지금**: "전환까지 걸리는 시간"을 지수분포 하나로 가정. 즉 "클릭 직후에 살 확률과 3일 뒤에 살 확률이 같다"는 뜻.

```
# MTLCrossv2dfm.forward_train
delay = torch.exp(torch.clamp(self.mainFF_delay(output), max=30))   # λ 하나 → 지수분포
```

**문제**: 실제로는 밤에 몰리고(24시간 주기), 고가 상품은 며칠 고민함. 지수분포로는 못 담음.

**바꿀 것**: Delay 헤드가 λ 하나 대신 **"1일차에 살 확률, 2일차에 살 확률, … 7일차"** $K$개를 softmax로 내놓게. 누적합(prefix-sum)이 단조 CDF가 된다 — TWICE의 delay head가 정확히 이 구조.

$$
S(d) = \prod_{k<d}(1-h_k), \qquad f(d) = S(d-1)\,h_d
$$

`timeunit=86400`, `max_delay=7`과 정확히 맞물림. 검열항은 `log(1 − p + p·S(d))`로 형태 유지, positive항은 `log p + log f(d)`. 24시간 주기까지 잡으려면 첫날을 시간 단위로 촘촘히(NoDeF의 커널 의사점).

**효과**: 미전환 샘플을 얼마나 믿을지가 정확해짐. A3의 전제.

→ **배치 DFM(구매)에만** 해당. 온라인 3종은 지연 분포 자체가 없음. loss와 헤드 차원만 바꾸고 파이프라인은 안 건드림.

### A2. λ를 가짜 미전환에서 보호 — stop-gradient + 입력 축소 (TWICE)

**지금**: 미전환 항 `log(1 − p + p·e^{−λd})`이 p와 λ를 **동시에** 밈. 아직 안 온 전환(가짜 미전환)이 λ를 왜곡.

**바꿀 것**:
- 미전환 항에서 λ로 가는 기울기를 끊음(`λ.detach()`) → λ는 **실제로 도착한 전환**(positive 항 `log λ − λd`)에서만 배움. TWICE 식 (9) $\hat p_o = p_\theta \cdot \text{sg}[F_\phi]$
- λ 입력을 전체 184차원 대신 **안정적인 몇 개 필드**(목적·캠페인·계정·지면·시간대)로만. 전체 $x$ 로 조건화하면 도착 조건부 목적함수가 퇴화한다는 TWICE §4.3 논거

**효과**: λ가 흔들리지 않음. 버킷별 지원 표본이 두꺼워짐.

→ 몇 줄. 단 우측 검열 우도에서 negative 항도 λ 정보를 담고 있어(생존분석 표준) "더 옳다"가 아니라 "더 강건하다"는 교환 — A/B로 판정.

### A3. 버려지는 λ로 임의 창 CVR 서빙 (PI)

> Zhang et al., *Personalized Interpolation*, CIKM 2025 — https://arxiv.org/abs/2501.14103

**지금**: λ를 학습만 하고 추론(`forward_without_postproc`)에서 버림. 서빙은 7일 창 하나. 그런데 vcvr은 `max_conv_delay: 1440`(1일), 나머지는 7일 — **목적마다 창이 다른데** 모델은 모름.

**바꿀 것**: `(p, λ)`를 함께 내보내고 서빙이 목적별 창 $T$ 를 넣어 계산. 지수분포 지연 가정(= DFM) 아래서는 보간조차 필요 없는 닫힌형.

$$
P(\text{T일 안에 전환} \mid x) = \hat p(x)\,\big(1 - e^{-\hat\lambda(x)\,T}\big)
$$

(A1의 버킷 CDF면 $\hat p(x)\cdot F(T)$.) PI의 지수형 보간 계수 $\alpha = \frac{e^{-\beta T_s} - e^{-\beta T_f}}{e^{-\beta T_s} - e^{-\beta T_l}}$ 가 이 식과 같다.

**효과**: **학습 변경 0**으로 1일·3일·7일 CVR을 한 모델에서. 광고주가 창을 고르게 하는 것(Meta의 FOW)도 같은 식. 덤으로 $\hat p(1-e^{-\hat\lambda\cdot 1})$ 과 실측 1일 전환율을 비교하면 **λ 헤드 캘리브레이션 진단이 공짜**.

→ 가장 싸다. A2로 λ가 믿을 만해진 뒤가 좋다.

## B. 관측 창·헤드 설계

**"얼마나 기다리고, 헤드를 어떻게 나누고, 어떻게 합치나."** A가 λ·검열항을 *잘 만들자*라면 B2는 λ·검열항을 *아예 안 쓰자*다 — 배치 라인에서 둘을 대조군으로 세우면 어느 쪽이 맞는지 한 실험으로 갈린다.

### B1. 온라인 모델에 "잠깐 기다리기" + 지연 보조 헤드 (ES-DFM)

> Yang et al., *Capturing Delayed Feedback in CVR Prediction via Elapsed-Time Sampling*, AAAI 2021 — https://arxiv.org/abs/2012.03245

**지금**: 클릭이 오면 **0초 기다리고** 바로 "전환 안 함"으로 학습. 모든 전환이 일단 오답으로 시작. (`_fnw`는 ES-DFM의 $e = 0$ 특수해)

**바꿀 것**: 1시간 정도 기다린 뒤 라벨 붙이기 + "나중에 전환할 확률" 헤드($f_{dp}$, $f_{rn}$) 추가. `_fnw` 가중치 `(1+p̂)`, `(1−p̂)(1+p̂)` → `(1+f_dp)`, `(1+f_dp)·f_rn`. 보조 헤드 라벨은 C1로 만든다.

**효과**: 가짜 미전환이 크게 줄고, 가중치가 $[0,1]$ 확률로만 만들어져 폭발하지 않음.

→ 온라인 3종(설치·친구추가·가입). 스트림 emit 시점을 바꿔야 해서 파이프라인 비용 있음. 데이터엔 이미 `conv_delay` 컬럼이 있어 $f_{dp}$ 라벨은 만들 수 있음.

### B2. 창별로 "확정된 라벨"만 써서 학습 (FTP · Defer-offline · DEFUSE)

> Li et al., *Follow the Prophet*, SIGIR 2021 — https://arxiv.org/abs/2108.06167
> Defer 오프라인 버전은 ESDF §3.2에 정리됨 — https://arxiv.org/abs/2308.04768
> Chen et al., *DEFUSE*, WWW 2022 — https://arxiv.org/abs/2202.06472

**지금**: 모든 헤드가 "아직 모르는" 샘플을 안고 배우고, 그걸 가중치나 λ로 사후 보정.

**바꿀 것**: 헤드를 "1시간 안에 전환 / 1일 안에 / 7일 안에" 식으로 나누고, **각 헤드는 그 시간이 이미 지난 샘플로만 학습**. 라벨이 전부 확정된 상태 → 가짜 미전환이 아예 없음. 합칠 때는 "7일 다 지난 데이터로 만든 완벽한 모델(**prophet**)"에 가장 가까운 헤드를 고르도록 aggregator $g(x)$ 를 학습(FTP 식 10·11) — 또는 B4(a)의 synthesizer.

**DEFUSE의 IP/DP 분리는 이것의 창 2개짜리 특수형**이다:

```
p(y=1|x) = F_IP(x) + F_DP(x)
F_IP: 창 안 전환 — 중복도 없고 라벨도 확정 → 순수 BCE, 보정 없음
F_DP: 창 밖 전환 — FNW(w_o=0)와 같은 편향 분포 → 여기에만 ES-DFM/DEFUSE 가중
```

깨끗한 신호(IP)가 지저분한 보정(DP)에 흔들리지 않는다. 배치 DFM에서도 IP 샘플은 검열항이 필요 없으니 그대로 적용.

**효과**: 보정 가중치·λ·검열항 전부 불필요. 순수 BCE. 분산 감소.

→ Alibaba가 실제로 쓰는 구조(Defer n+1 tower). `MTLSimpledfom`·`MTLCrossv2dfm`이 이미 action_type별 T헤드 구조라 **헤드를 K배로 늘리고 합산 규칙만** 바꾸면 됨. `dataset_batch_cvr_v3`에 `clickts/convts`가 있어 창별 라벨 생성이 한 줄.

### B3. 진짜 미전환도 다시 넣기 (DEFER)

> Gu et al., *Real Negatives Matter*, KDD 2021 — https://arxiv.org/abs/2104.14121

**지금**: 전환한 샘플만 나중에 한 번 더 넣음(추정). 그래서 전환 잘 되는 피처 영역만 데이터가 부풀려짐 ($q(x) \ne p(x)$).

**바꿀 것**: 어트리뷰션 창(7일) 지나면 **모든 클릭**을 확정 라벨로 한 번 더 넣기.

**효과**: 데이터 왜곡 제거 + "확실히 안 산다"는 정보가 생김(지금 온라인 모델엔 없음). 가중치도 단순해짐 — FNW 변형이면 positive 가중치 **상수 2**, FNC면 `2q`.

→ 데이터 2배가 비용.

### B4. 헤드 합산을 "고정 mask 합" 대신 학습으로 (MISS · MAC · NSN/DDFM)

> Liu et al., *MISS*, AAAI 2024 — https://ojs.aaai.org/index.php/AAAI/article/view/28726
> Wu et al., *MAC: Multi-Attribution Benchmark*, KDD 2026 — https://arxiv.org/abs/2603.02184
> Liu et al., *Neural Satellite Networks*, SIGIR 2023 — https://doi.org/10.1145/3539618.3591747 · Dai et al., *DDFM*, CIKM 2023 — https://doi.org/10.1145/3583780.3614856 (두 편은 ACM 유료라 초록 기준)

**지금**: `output_mask[objective]`로 헤드를 **고정 비율**로 더함. 어느 헤드가 지금 더 믿을 만한지, 어느 라벨 소스가 주인지 반영 안 됨.

세 논문이 같은 자리를 다른 축에서 고친다:

- **(a) 창별 헤드 → synthesizer (MISS)**: 헤드 출력들(+정규화 버전)을 입력으로 받는 **2층 MLP + softmax**가 동적 가중치. 학습 데이터는 **assembled pipeline** = 최신 확정 전환 ∪ 7일 지난 확정 미전환 — 옛 positive를 최신 positive로 갈아 끼우면 이상적 분포에 더 가깝다(MISS Fig. 3). 기존 출력 위에 그냥 얹을 수 있어 **캘리브레이터로도** 쓸 수 있음.
- **(b) 라벨 소스별 헤드 → 비대칭 전이 (MAC)**: `OBJECTIVE_MAP["PURCHASE"] = ["bcon_Purchase", "Purchase"]` — 두 라벨 소스 헤드를 지금은 단순 합산. MAC의 발견: 여러 어트리비션 라벨을 함께 배우면 좋아지지만 **보조가 주를 오염시키면 오히려 나빠짐**(first-click 실험). 과금 기준 라벨을 주 헤드로, 나머지는 보조로 두고 **보조 → 주 한 방향으로만** 전이(MoAE). 1일 창 vs 7일 창 라벨에도 같이 적용 가능.
- **(c) 배치 × 온라인 두 모델 → 같은 synthesizer (NSN · DDFM)**: 구매는 배치(정확·느림), 설치/가입/친구추가는 온라인(신선·거침)으로 **목적별로 하나만** 서빙 중. 같은 목적에 둘 다 돌리고 (a)로 합치면 NSN(긴 창 main + 짧은 창 satellite)·DDFM(스트림 추정기 + 확정 라벨 추정기) 구조가 된다. 두 라인 인프라가 이미 있어 실험 비용 낮음.

→ (a)는 가장 싼 항목이고 B2·(c)의 전제. (b)는 헤드가 이미 분리돼 있어 합산 규칙만 바꾸면 됨.

## C. 라벨 교정

**"미전환 샘플에 가중치를 깎는 대신, 라벨 자체를 고친다."** 세 항목이 한 세트다 — C1이 학습 데이터를 만들고, C2가 교정 모델을 세우고, C3가 발산을 막는다.

### C1. 과거 로그로 미래 라벨 만들기 (FSIW · 대안 DLA-DF)

> Yasui et al., *A Feedback Shift Correction*, WWW 2020 — https://arxiv.org/abs/2002.02068
> Saito, Morishita & Yasui, *Dual Learning Algorithm for Delayed Conversions*, SIGIR 2020 — https://arxiv.org/abs/1910.01847

B1·B2·C2에 필요한 "이 클릭이 창 안에 전환할까 / 나중에 뒤집힐까" 라벨을 어떻게 구하나?

**방법**: 지금 로그의 시계를 $\tau$(예: 7일) 전으로 되감아 "그때 기준으로 창 안이었나"를 표시(counterfactual deadline). 기다릴 필요 없이 지금 있는 데이터로 보조 헤드 학습셋을 만듦.

**대안 (DLA-DF)**: 되감기 없이 propensity $\theta(X,E)$ 모델을 CVR 모델과 교대로 학습(ICVR 추정기). 합성 데이터 실험만 있어 근거 약함 — counterfactual 데이터가 부족할 때만.

→ 배치 잡 하나. B1·B2·C2의 전제 조건.

### C2. 미전환에 "전환 확률" 라벨을 붙여서 학습 (ULC)

> Wang et al., *Unbiased Delayed Feedback Label Correction*, KDD 2023 — https://arxiv.org/abs/2307.12756

**지금**: DFM 검열항은 "관측된 라벨"이 나올 확률을 최대화. 진짜(oracle) 라벨 기준이 아님.

**바꿀 것**: 보조 모델 $g(x, e)$ 가 미전환 샘플마다 "이건 나중에 $w$ 확률로 전환"이라고 라벨 붙임.

$$
\mathcal{L}_{LC} = v\log f + w(1-v)\log f + (1-w)(1-v)\log(1-f)
$$

보조 모델은 **CVR 모델의 임베딩을 복사해서 시작** → 데이터 적어도 잘 배움. 둘을 번갈아 학습(Algorithm 1). 학습 데이터는 C1.

**효과**: Theorem 1 — $w$ 가 정확하면 oracle 손실의 불편 추정. 신규 그룹에도 잘 먹힘(`# C1 개선안 제안` §1 참고).

→ 지금 delay 헤드 자리에 헤드 하나 추가하면 임베딩 공유는 자동.

### C3. 손실이 음수로 내려가지 않게 막기 (nnDF)

> Kato & Yasui, *Learning Classifiers under Delayed Feedback with a Time Window Assumption*, KDD 2022 — https://arxiv.org/abs/2009.13092

**지금**: `loss.py`의 `puloss`처럼 positive에서 negative 항을 **빼는** 손실이 있음. 이런 합성 손실은 모델이 크면 **경험 위험이 0 아래로 내려가 발산**할 수 있음. 논문에서 보정 없는 `convDF`가 Criteo에서 실제로 발산.

**바꿀 것**: 음수 부분에 `max(0, ·)` (nnPU 방식). `puloss`, C2의 LC 손실, `fnw_neg` 최초 버전(마이너스)처럼 **빼는 항이 있는 손실 전부** 해당.

→ C2를 도입하면 필수 안전장치. 손실 함수 두 줄.

nnDF 자체는 **λ 없이 시간창 + 전 샘플로 볼록·불편 위험**을 만드는 방법이라, 배치 라인에서 DFM의 지연 분포 가정을 통째로 뺀 대조군으로도 쓸 수 있음.

## D. post-click 행동 신호

### D1. 장바구니를 "구매 예고 신호"로 (GDFM) → 행동 궤적으로 (TRACE)

> Yang & Zhan, *Generalized Delayed Feedback Model with Post-Click Information*, NeurIPS 2022 — https://arxiv.org/abs/2206.00407
> Zhang, Ding & Ao, *Follow the TRACE*, SIGIR 2026 — https://arxiv.org/abs/2604.23197

**지금**: 장바구니 담기는 클릭 후 **몇 분**, 구매는 **며칠** 걸림. 그런데 장바구니 이벤트가 구매 예측을 전혀 안 건드림. `OBJECTIVE_MAP`에 CART 액션(`bcon_AddToCart`, `bcon_AddToWishList`, `bcon_ViewCart`)이 있지만 `mtldfm_v2`는 positive가 자기 action_type 헤드만 갱신.

**1단계 — GDFM**: "장바구니에 담은 사람 중 몇 %가 사는지"($q_\phi(a \mid x, y, \delta)$)를 배워서, 장바구니가 관측되면 **그 순간 구매 확률을 미리 올려놓기**.

$$
\mathcal{L}'_{\delta} = -\log \sum_{y} q_\phi(a \mid x, y, \delta)\, q_\theta(y \mid x), \qquad w = e^{-\alpha H(y|a)} \cdot e^{-\beta\delta}
$$

가중치 = 정보량(조건부 엔트로피) × 신선도. KL 정규화로 지연 모델에서 너무 벗어나지 않게. Taobao 실험에서 $p(a{=}1 \mid y{=}1)$ 이 CVR보다 훨씬 안정적이라 대리로 쓸 만하다는 근거.

**2단계 — TRACE**: 행동을 하나씩 보지 않고 **"10분 뒤 장바구니 → 1시간 뒤에도 그대로 → 1일 뒤 미구매"** 같은 누적 궤적 $\xi$ 전체를 본다.

$$
p(y \mid x, \xi) = \text{Softmax}_y\big[\log p_\theta(y \mid x) + \textstyle\sum_h \alpha_h \log p_\psi(o_h \mid x, y)\big]
$$

- $p_\psi(o_h \mid x, y)$("전환하는 클릭이면 $h$ 시점 상태가 이럴 확률")는 **배치 라인**(7일 성숙 로그)에서 배워 고정
- $p_\theta$ 는 **온라인 라인**에서 갱신. 미확정 샘플엔 hard label 대신 관측 창의 주변 우도
- 초기 궤적이 비어 있을 때는 전 생애 로그를 랜덤 절단해 학습한 completer가 보완(신뢰도 게이트)

**효과**: 구매 pCVR이 며칠 뒤가 아니라 몇 분 뒤에 반응. Alibaba 로그에서 **구매의 50%가 직전 행동 1시간 내**.

→ 장바구니 데이터는 이미 있음. 연결만 없음. TRACE는 클릭별 post-click 상태 컬럼(10분·1시간·1일)이 필요하고, 두 라인 역할 분담이 자연스러움.

## E. 구매 모델의 배수 축 (mpc)

mpc 출력 = pCVR × `conv_multiplier`. 이 그룹은 그 **배수를 위(재구매)와 아래(환불)에서 각각** 고친다. READER·TESLA·ESDF는 같은 Alimama 팀 형제 논문이다.

### E1. 재구매 배수를 그룹 스칼라 대신 클릭별 예측으로 (READER)

> Li et al., *Delayed Feedback Modeling for Post-Click GMV Prediction*, WWW 2026 — https://arxiv.org/abs/2601.20307

**지금**: 배수는 redicoke 문서에서 오는 **group_id별 정적 스칼라**. `AdConvMultiplierModel`은 전환된 클릭만으로 오프라인 회귀. 새 그룹은 1.0(`delayed_feedback_task.md` C4 참고).

**바꿀 것** (Taobao: 전환 클릭의 **53.55%가 재구매**):
- **Router** $r = \sigma(f_\phi(x)/T)$: 이 클릭이 2회 이상 구매로 이어질 확률. 예측기와 **분리된** 네트워크(기울기 얽힘 방지)
- **Dual tower**: 단일구매 / 재구매 타워, 3구역 라우팅($r \le 0.1$ 단일, $\ge 0.9$ 재구매, 사이는 $(1-r)\hat y_s + r\hat y_r$)
- **Label Calibrator**: 창이 닫히기 전 관측 누적값 $y^{(t)}$ 는 과소 → $\hat\delta = \text{softplus}(f_\psi(x, \Delta t, N_t))$ 로 $\log(1+y^*) - \log(1+y^{(t)})$ 를 예측해 pseudo-label 생성 → **온라인 학습 가능**
- **GRA**: 창이 닫히면 진짜 라벨로 다시 맞춤

**효과**: 재구매 이질성을 클릭 단위로 잡고, 7일 기다리지 않고 배수 갱신. 신규 그룹 배수=1.0도 router가 피처로 예측하므로 함께 풀림.

→ 배수 대상을 "구매 건수"로 두면 지금 econv 파이프라인에 그대로 들어감.

### E2. 환불 빼기 — ECVR (ESDF · TESLA)

> Zhao et al., *Entire Space Cascade Delayed Feedback Modeling*, CIKM 2023 — https://arxiv.org/abs/2308.04768
> Luo et al., *Modeling Cascaded Delay Feedback for Online NetCVR*, WWW 2026 — https://arxiv.org/abs/2601.19965

**지금**: mpc = pCVR × 재구매 배수. **환불은 안 뺌**.

**바꿀 것**: 전환 샘플 위에 환불률(RFR) 헤드 + 환불 어트리뷰션 창 → $p_{ecvr} = p_{cvr} \times (1 - p_{rfr})$. RFR은 전환 샘플에만 라벨이 있어 SSB·DS가 심하니 ESMM식으로 CVRFR($p(y{=}1, z{=}0 \mid x)$) 태스크를 전 클릭 공간에서 함께 학습(ESDF). 온라인이면 TESLA의 스트림 설계 — 전환 뒤 **환불 관측 창**을 따로 두고, 두 단계 각각에 ES-DFM형 중요도 가중($w^+ = 1 + p\cdot P(h > W_{obs})$)을 stage-wise로 적용.

**효과**: Alibaba A/B에서 ECVR +5.21%(ESDF), NetCVR RI-AUC +12.41%p(TESLA).

→ **환불 로그가 있어야 함.** 확인 먼저.

## F. 학습·서빙 기타

### F1. FNC 한번 켜보기 (Ktena)

> Ktena et al., *Addressing Delayed Feedback for Continuous Training*, RecSys 2019 — https://arxiv.org/abs/1907.06558

코드에 이미 있는데(`forward_recalibration`) 전부 꺼져 있음. 가중치 대신 추론 때 `q/(1−q)`로 보정하는 방식.

**효과**: 성능은 논문상 FNW와 비슷. 가중치가 모델 자기 예측(`y_hat.detach()`)에 의존하는 구조를 없앨 수 있음.

→ config 한 줄(`recalibration: true` + `loss_info.type: BCELoss`). 한 라인에서 A/B만 해볼 가치. **FNW와 동시에 켜면 이중 보정**이므로 반드시 택일.

### F2. 즉시 전환과 늦은 전환을 다르게 — 지연 인식 ranking 보조 손실 (TESLA)

**지금**: 손실이 전부 pointwise(BCE/NLL). 클릭 직후 전환과 6일 뒤 전환이 같은 무게의 positive.

**바꿀 것**: pairwise ranking 항을 **작은 계수로** 추가 — 지연에 따라 positive 가중, 불확실한 negative 위주로 샘플링(온도 $\tau$).

**주의**: ranking loss는 AUC를 올리지만 **캘리브레이션을 흔들 수 있음.** 입찰가에 직결되니 RIG·cal 감시하에 보조로만.

### F3. 재학습 없이 늦은 전환 반영 — influence function (IF-DFM)

> Ding et al., *Delayed Feedback Modeling with Influence Functions*, AAAI 2026 — https://arxiv.org/abs/2502.01669

**지금**: 배치 mpc는 늦게 도착한 전환(라벨 반전)을 **4시간마다 20일치 전체 재학습**으로만 흡수.

**바꿀 것**: 라벨 반전을 데이터 섭동으로 보고 파라미터 변화량을 직접 계산

$$
\Delta\theta \approx -\tfrac{1}{n} H^{-1}\Big[\textstyle\sum_{j\in J}\big(\nabla L(z_j^\delta) - \nabla L(z_j)\big) + \sum_{k\in K}\nabla L(z_k)\Big]
$$

$H^{-1}$ 을 직접 안 구하고 $\min_\Delta \tfrac12\Delta^\top H\Delta - \langle b, \Delta\rangle$ 를 **SGD로 푸는 finite-sum 문제**로. 보조 모델 없음.

**효과**: 전체 재학습은 일 1회, 사이엔 가벼운 갱신 → 배치 라인 신선도 ↑. C3(갱신량)의 정공법.

→ DCN + 대형 임베딩의 Hessian이라 **비용 가장 큼**. 마지막에.

### F4. 소재 멀티모달 임베딩 (서베이 §6.3.1·6.3.4)

> Xue, Yang & Zhai, *Conversion rate prediction in online advertising: … future directions*, 2025 — https://arxiv.org/abs/2512.01171

6개 서빙 config에 **소재 피처가 하나도 없음**(`image_hash`는 vcvr만). 서베이 1순위 미래 방향이 LLM/멀티모달 소재 표현. 소재 교체가 모델에 안 보이는 문제(설정변경 anchoring)도 같이 풀림.

→ 피처 파이프라인 작업.

## 순서 (전체)

| 순서 | 항목 | 이유 |
|---|---|---|
| **1** | A3 λ로 임의 창 서빙 | **학습 0**. λ 진단도 공짜 |
| 2 | B4(a) synthesizer | 가장 싼 모델. B2·B4(c)의 기반, 캘리브레이터로도 |
| 3 | A1 + A2 버킷 CDF + λ 보호 | loss·헤드만. 파이프라인 안 건드림. A3의 신뢰도 |
| 4 | C1 → B2 라벨 생성 → 창별 확정 헤드 | 가짜 미전환을 학습에서 제거. MTL 골격 재활용 |
| 5 | B4(b) 비대칭 합산 | 헤드 이미 분리됨. 합산 규칙만 |
| 6 | D1(1단계) 장바구니 신호 | 효과 가장 클 가능성. 데이터 있음 |
| 7 | C3 → C2 음수 방지 → 라벨 교정 | 배치 검열항 대체 |
| 8 | E1 재구매 router | econv 대체. 새 그룹 배수=1.0 해소 |
| 9 | B1 ES-DFM 대기 창 | 파이프라인 변경 |
| 10 | B3 DEFER 재투입 | 데이터 2배 |
| 11 | D1(2단계) TRACE | post-click 상태 컬럼 필요 |
| 12 | F2 ranking 보조 | cal 감시 필수 |
| 13 | F4 소재 임베딩 | 피처 파이프라인 |
| 조건부 | E2 환불 | 로그 확인 후 |
| 언제든 | F1 FNC · B4(c) 배치×온라인 | config 한 줄 · 인프라 있음 |
| 마지막 | F3 IF-DFM | 비용 최대 |

> **대조군 설계**: A(λ·검열항을 잘 만들자)와 B2(λ·검열항을 안 쓰자)는 배치 라인에서 서로의 대조군이다. 같은 실험에서 "지연 분포를 모델링하는 게 낫나, 창별 확정 라벨이 낫나"가 갈린다.

## 논문 외, 코드에서 바로 고칠 것

- delay 하한 `1e-5`일(0.86초) → 1분 정도로. 즉시 전환이 `log λ` 항을 통해 λ를 위로 밀어올림
- `lr_scheduler`의 `lr_decay()`가 상수 0.95를 반환 → lr × 0.95 고정이라 스케줄이 아님
- Clippy `lambda_abs`(0.01), `lambda_rel`(0.5)이 하드코딩 → `build_optimizers`에서 config로 노출
