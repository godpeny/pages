# Delayed Feedback
## Modeling Delayed Feedback in Display Advertising
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