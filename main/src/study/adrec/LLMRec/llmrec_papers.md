# 광고추천 LLM 백본 도입을 위한 논문 목록

## 배경

광고추천 업무에 LLM 모델을 백본으로 도입하기 위해 관련 논문을 조사합니다.
선정 기준은 **인용수**, **출처 기관**(빅테크, 유명 대학), **학회·학술지 게재 여부** 세 가지입니다.

### 출처

| 출처 | 내용 |
|---|---|
| 자체 조사 | 기준에 맞는 논문을 직접 검색 |
| 아지트 "추천 스터디" | louie.kim의 2024~2026년 글 56개에서 뽑은 논문 중 LLM·생성형 백본 관련 논문 |
| 위키 `study.` 하위 페이지 | 위 스터디 논문 중 직접 자세히 스터디한 논문 (표의 ✓) |
| 노션 서베이 | "대규모 언어 모델 기반 추천 시스템: LLM의 역할과 아이템 표현에 따른 서베이"의 참고문헌 |

### 읽는 법

- 인용수와 게재처는 Semantic Scholar 조회값입니다 (조회일 2026-11-26 ~ 2026-11-27).
- 소속 기관은 원문 확인 없이 기재한 것이 많습니다. 스터디 메모에서 가져온 추정은 "(메모 기준)"으로 표기했습니다.
- 온라인 A/B 성과는 모두 각 회사의 자체 보고 수치입니다.
- **스터디 ✓** 는 위키에서 직접 스터디한 논문입니다. 이 논문들은 대부분 2026년 논문이라 인용수 기준을 적용하지 않고 "기관 + 직접 검토"로 포함했습니다.

광고 도메인에서 실제 서빙까지 간 논문은 대부분 2025~2026년 것이라 인용수가 아직 낮고, 인용수가 높은 논문은 대부분 2022~2023년 학계 기초 연구입니다.
그래서 인용수 하나로 줄 세우지 않고 **광고 직접 관련성**과 **인용수**를 나눠서 봅니다.

## "LLM 백본"의 세 가지 의미

논문마다 뜻이 달라서, 어느 쪽을 도입하려는지에 따라 읽을 목록이 달라집니다.

| 유형 | 설명 | 대표 논문 |
|---|---|---|
| **A. 사전학습 LLM을 그대로 백본으로 사용** | 텍스트로 학습된 LLM을 추천 데이터로 파인튜닝 | HLLM, 360Brew, PLUM, LEADRE, GenRec, NEO |
| **B. LLM식 구조를 행동 로그로 처음부터 학습** | 트랜스포머·생성형 구조만 빌리고 언어 지식은 쓰지 않음 | HSTU, OneRec, GPR, LLaTTE |
| **C. LLM을 보조 신호로 사용** | 기존 CTR 모델은 유지하고 LLM 임베딩·지식을 피처로 투입 | KAR, CTRL, BAHE, LLM-ESR |

노션 서베이는 LLM의 역할을 Enhancer, Recommender(텍스트 생성 / 카탈로그 점수화 / 시맨틱 ID 생성), Agent, Simulator로 나눕니다.
Enhancer는 유형 C, Recommender는 유형 A·B와 겹치고, Agent와 Simulator는 [6장](#6-llm-as-agent--simulator--설명)에 따로 모았습니다.

---

## 1. 광고 시스템에 직접 적용한 논문

| 논문 | 기관 | 게재처 | 인용 | 스터디 | 내용 |
|---|---|---|---|---|---|
| [GPR](https://arxiv.org/abs/2511.10138) (2025) | Tencent, Tsinghua | KDD | 48 | | 검색·랭킹 캐스케이드를 단일 생성 모델로 대체, Weixin Channels 광고 전면 배포 |
| [GR4AD](https://arxiv.org/abs/2602.22732) (2026) | Kuaishou | KDD | 31 | | 대규모 광고용 실시간 생성 추천 |
| [ExFM](https://arxiv.org/abs/2502.17494) (2025) | Meta | WWW | 28 | | 초대형 파운데이션 모델을 광고에 서빙하는 증류 구조 |
| [EGA-V2](https://arxiv.org/abs/2505.17549) (2025) | Meituan | arXiv | 28 | | 광고 생성·배치·과금까지 엔드투엔드 |
| [LLaTTE](https://arxiv.org/abs/2601.20083) (2026) | Meta | KDD | 22 | | 광고 시퀀스 모델링의 스케일링 법칙 |
| [RARE](https://arxiv.org/abs/2504.01304) (2025) | Tencent | EMNLP | 18 | | LLM이 상업 의도를 생성해 검색광고 리트리벌 |
| [LEADRE](https://arxiv.org/abs/2411.13789) (2024) | Tencent | VLDB | 12 | | LLM 기반 디스플레이 광고 생성 리트리벌 |
| [Meta Lattice](https://arxiv.org/abs/2512.09200) (2025) | Meta | KDD | 8 | | 광고 추천 모델 공간 재설계 |
| [LFM4Ads](https://arxiv.org/abs/2508.14948) (2025) | Tencent | CIKM | 7 | | 광고용 대형 파운데이션 모델 |
| [Taiji](https://arxiv.org/abs/2606.03866) (2026) | Kuaishou | arXiv | 2 | | LLM 보조 신호 방식, 광고 플랫폼 배포 |
| [LLM Retrieval for Ad Recommendations](https://arxiv.org/abs/2605.21969) (2026) | Meta | arXiv | 0 | ✓ | LLM 기반 광고 후보 생성으로 안정성·예측 가능성 확보 |

논문은 아니지만 Meta의 GEM 엔지니어링 블로그도 함께 볼 만합니다.

- [GEM (2025-11)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/): 인스타그램 광고 전환 5%, 페이스북 피드 3% 증가 보고
- [GEM 후속 (2026-10)](https://engineering.fb.com/2026/10/13/ml-applications/metas-generative-ads-model-gem-scaling-the-central-model-and-compute-engine-for-frontier-ads-ai): 단일 중앙 모델과 Kunlun 구조

## 2. 산업 추천 시스템의 LLM·생성형 백본 (광고 외)

| 논문 | 기관 | 게재처 | 인용 | 스터디 | 내용 |
|---|---|---|---|---|---|
| [TIGER](https://arxiv.org/abs/2305.05065) (2023) | Google | NeurIPS | 558 | | Semantic ID 기반 생성 리트리벌의 출발점 |
| [HSTU](https://arxiv.org/abs/2402.17152) (2024) | Meta | ICML | 436 | ✓ | 조 단위 파라미터 생성 추천 |
| [OneRec](https://arxiv.org/abs/2502.18965) (2025) | Kuaishou | arXiv | 407 | ✓ | 검색·랭킹 통합 생성 모델 |
| [Wukong](https://arxiv.org/abs/2403.02545) (2024) | Meta | ICML | 188 | | 추천 스케일링 법칙 (GEM의 기반 구조) |
| [MTGR](https://arxiv.org/abs/2505.18654) (2025) | Meituan | CIKM | 146 | | 산업 규모 생성 추천 프레임워크 |
| [HLLM](https://arxiv.org/abs/2409.12740) (2024) | ByteDance | arXiv | 143 | | 아이템 LLM + 유저 LLM 계층 구조 |
| [Semantic ID 랭킹](https://arxiv.org/abs/2306.08121) (2023) | Google | RecSys | 140 | | 랭킹에서 Semantic ID 일반화 |
| [LONGER](https://arxiv.org/abs/2505.04421) (2025) | ByteDance | RecSys | 139 | | 초장기 시퀀스 모델링 |
| [Long Semantic IDs in Parallel](https://arxiv.org/abs/2506.05781) (2025) | Meta 산학 (메모 기준) | KDD | 137 | | 긴 Semantic ID 병렬 생성 |
| [RankMixer](https://arxiv.org/abs/2507.15551) (2025) | ByteDance | CIKM | 131 | | 랭킹 모델 스케일업 |
| [COBRA](https://arxiv.org/abs/2503.02453) (2025) | Baidu | NeurIPS (확인 필요) | 104 | | 희소 ID + 밀집 벡터 결합 생성 |
| [OneRec-Think](https://arxiv.org/abs/2510.11639) (2025) | Kuaishou | ACL | 104 | | 생성 추천에 텍스트 내 추론 결합 |
| [TWIN V2](https://arxiv.org/abs/2407.16357) (2024) | Kuaishou | CIKM | 92 | | 초장기 행동 시퀀스 CTR 모델링 |
| [GRID 핸드북](https://arxiv.org/abs/2507.22224) (2025) | Snap | CIKM | 78 | | Semantic ID 생성 추천 실무 가이드, 오픈소스 |
| [PLUM](https://arxiv.org/abs/2510.07784) (2025) | Google/YouTube | WWW (확인 필요) | 76 | | 사전학습 LLM을 산업 생성 추천에 적응 |
| [LEARN](https://arxiv.org/abs/2405.03988) (2024) | Kuaishou | AAAI | 59 | | LLM 지식을 추천으로 이전 |
| [Semantic IDs for Joint Search and Rec](https://arxiv.org/abs/2508.10478) (2025) | Spotify | RecSys | 48 | | 검색·추천 공용 Semantic ID |
| [Semantic ID 임베딩 안정성](https://arxiv.org/abs/2504.02137) (2025) | Meta | RecSys | 47 | | Semantic ID로 임베딩 표현 안정화 |
| [OnePiece](https://arxiv.org/abs/2509.18091) (2025) | Shopee | KDD | 46 | | 캐스케이드 랭킹에 컨텍스트 엔지니어링·추론 도입 |
| [LIGER](https://arxiv.org/abs/2411.18814) (2024) | Meta | TMLR | 44 | | 생성 리트리벌과 밀집 리트리벌 통합 |
| [OneRec Technical Report](https://arxiv.org/abs/2506.13695) (2025) | Kuaishou | arXiv | 37 | | OneRec 전체 시스템 기술 보고서 |
| [HyFormer](https://arxiv.org/abs/2601.12681) (2026) | ByteDance | SIGIR | 35 | ✓ | 시퀀스 모델링과 피처 인터랙션 통합 |
| [OneRec-V2](https://arxiv.org/abs/2508.20900) (2025) | Kuaishou | arXiv | 34 | | OneRec 후속 버전 |
| Make It Long, Keep It Fast (2026) | ByteDance (Douyin) | WWW | 34 | | 10k 길이 시퀀스 엔드투엔드 모델링 |
| [LUM](https://arxiv.org/abs/2502.08309) (2025) | Alibaba | WSDM | 33 | | 3단계 대형 유저 모델 |
| [GRAM](https://arxiv.org/abs/2504.01403) (2025) | JD.com | WWW | 30 | | 이커머스 생성 리트리벌·정렬 |
| [STATIC (Vectorizing the Trie)](https://arxiv.org/abs/2602.22647) (2026) | Google/YouTube | KDD | 26 | ✓ | LLM 생성 리트리벌의 제약 디코딩 가속 |
| [TransAct V2](https://arxiv.org/abs/2506.02267) (2025) | Pinterest | CIKM | 26 | | 평생 행동 시퀀스 모델링 |
| [LiRank](https://arxiv.org/abs/2402.06859) (2024) | LinkedIn | KDD | 26 | | 산업 대규모 랭킹 모델 |
| [DV365](https://arxiv.org/abs/2506.00450) (2025) | Meta | KDD | 19 | | 인스타그램 초장기 유저 이력 임베딩 |
| [VISTA (Massive Memorization)](https://arxiv.org/abs/2510.22049) (2025) | Meta, Yale | ICLR 2026 | 18 | ✓ | 수백 조 파라미터 생성 추천의 산업 확장성 |
| [MixFormer](https://arxiv.org/abs/2602.14110) (2026) | ByteDance | KDD | 18 | | dense와 시퀀스를 함께 스케일업 |
| [360Brew](https://arxiv.org/abs/2501.16450) (2025) | LinkedIn | arXiv | 15 | | 디코더 전용 LLM 하나로 랭킹 전체 대체 |
| [URM](https://arxiv.org/abs/2502.03041) (2025) | Alibaba | CIKM | 13 | | LLM을 범용 리트리버로 |
| [LEMUR](https://arxiv.org/abs/2511.10962) (2025) | ByteDance | CIKM | 12 | | 엔드투엔드 멀티모달 추천 |
| [GLIDE](https://arxiv.org/abs/2603.17540) (2026) | Spotify | arXiv | 9 | ✓ | Semantic ID 생성 리트리벌 팟캐스트 배포 |
| [NEO](https://arxiv.org/abs/2603.17533) (2026) | Spotify | arXiv | 7 | ✓ | 사전학습 LLM 하나로 검색·추천·추론 통합 |
| [LoopFM](https://arxiv.org/abs/2605.29280) (2026) | Meta | arXiv | 6 | ✓ | 파운데이션 모델의 과거 표현을 서빙 모델로 전달 |
| [MixLM](https://arxiv.org/abs/2512.07846) (2025) | LinkedIn | KDD | 6 | | LLM 랭킹 고처리량 서빙 |
| [Farewell to Item IDs](https://arxiv.org/abs/2601.22694) (2026) | ByteDance | arXiv | 5 | ✓ | 대형 랭킹 모델에서 ID를 시맨틱 토큰으로 대체 |
| [TokenMinds](https://arxiv.org/abs/2606.25147) (2026) | Google/YouTube | arXiv | 5 | ✓ | 사전학습 유저 토큰·임베딩 |
| [SORT](https://arxiv.org/abs/2603.03988) (2026) | Alibaba | CIKM | 5 | ✓ | 산업 규모 랭킹 트랜스포머 |
| [Rec-Distill](https://arxiv.org/abs/2605.29755) (2026) | ByteDance | arXiv | 4 | ✓ | 대형 추천 모델 증류 파이프라인 |
| [IAT](https://arxiv.org/abs/2604.08933) (2026) | ByteDance | arXiv | 4 | ✓ | 인스턴스를 토큰으로 압축한 시퀀스 모델링 |
| [Token Factory](https://arxiv.org/abs/2606.19635) (2026) | Google | RecSys | 3 | ✓ | 대형 추천 모델에 다양한 신호 통합 |
| [Feed SR](https://arxiv.org/abs/2602.12354) (2026) | LinkedIn | CIKM | 3 | ✓ | 피드 랭킹용 산업 규모 시퀀셜 추천 |
| [PrefixMem (LLMs Need Encoders)](https://arxiv.org/abs/2606.00324) (2026) | Pinterest | arXiv | 3 | ✓ | LLM의 Semantic ID 입력 인코더 |
| [RecGPT-Mobile](https://arxiv.org/abs/2605.04726) (2026) | Alibaba (Taobao) | SIGIR | 2 | ✓ | 온디바이스 LLM 유저 의도 이해 |
| [GenRec](https://arxiv.org/abs/2608.10257) (2026) | Netflix | arXiv | 미확인 | | 사내 파운데이션 LLM을 후학습한 전체 카탈로그 랭커 ([상세](#genrec-netflix-2026)) |

TWIN V2, TransAct V2, DV365, LiRank, LONGER, RankMixer는 LLM을 쓰지 않는 시퀀스·랭킹 모델입니다. LLM식 스케일업의 비교 대상으로 포함했습니다.

## 3. 학계 연구

### 3.1 기초 연구 (인용수 상위)

| 논문 | 기관 | 게재처 | 인용 |
|---|---|---|---|
| [P5](https://arxiv.org/abs/2203.13366) (2022) | Rutgers | RecSys | 1,143 |
| [TALLRec](https://arxiv.org/abs/2305.00447) (2023) | USTC, NUS | RecSys | 982 |
| [LLM은 제로샷 랭커](https://arxiv.org/abs/2305.08845) (2023) | Renmin Univ., UCSD | ECIR | 711 |
| [LC-Rec](https://arxiv.org/abs/2311.09049) (2023) | Renmin Univ. | ICDE | 534 |
| [RLMRec](https://arxiv.org/abs/2310.15950) (2023) | HKU | WWW | 520 |
| [Recformer](https://arxiv.org/abs/2305.13731) (2023) | UCSD, Amazon | KDD | 445 |
| [UniSRec](https://arxiv.org/abs/2206.05941) (2022) | Renmin Univ. | KDD | 445 |
| [InstructRec](https://arxiv.org/abs/2305.07001) (2023) | Renmin Univ. | ACM TOIS | 444 |
| [M6-Rec](https://arxiv.org/abs/2205.08084) (2022) | Alibaba | arXiv | 328 |
| [CoLLM](https://arxiv.org/abs/2310.19488) (2023) | USTC | IEEE TKDE | 269 |
| [BIGRec](https://arxiv.org/abs/2308.08434) (2023) | USTC, NUS | ACM TORS | 238 |
| [LLaRA](https://arxiv.org/abs/2312.02445) (2023) | USTC, NUS | SIGIR | 200 |
| [Pre-train, Align, and Disentangle](https://arxiv.org/abs/2412.04107) (2024) | Tencent 산학 (메모 기준) | SIGIR | 15 |
| [RecGPT](https://arxiv.org/abs/2506.06270) (2025) | HKU | EMNLP | 13 |

### 3.2 생성형 추천 개별 기법과 분석 (직접 스터디)

Semantic ID 생성, 디코딩, 학습 안정성처럼 생성형 백본을 실제로 만들 때 부딪히는 개별 문제를 다룬 논문입니다.

| 논문 | 기관 | 게재처 | 인용 | 내용 |
|---|---|---|---|---|
| [Rank-GRPO](https://arxiv.org/abs/2510.20150) (2025) | Cornell, Netflix | ICLR 2026 | 21 | LLM 대화형 추천을 강화학습으로 정렬 |
| [How Well Does GR Generalize?](https://arxiv.org/abs/2603.19809) (2026) | 미확인 | arXiv | 10 | 생성형 추천의 암기 대 일반화 분석 |
| [R3-VAE](https://arxiv.org/abs/2604.11440) (2026) | ByteDance | arXiv | 6 | Semantic ID 학습 안정화와 품질 지표 |
| [Catalog-Native LLM](https://arxiv.org/abs/2510.05125) (2025) | Roblox, Maryland, Cambridge | ICLR 2026 | 5 | 아이템 ID를 별도 방언으로 다루는 MoE 구조 |
| [Token-Efficient Item Representation via Images](https://arxiv.org/abs/2503.06238) (2025) | KAIST, UCSD | ICLR 2026 | 3 | 이미지로 아이템을 표현해 토큰 절약 |
| [IntRR](https://arxiv.org/abs/2602.20704) (2026) | Alibaba | arXiv | 3 | Semantic ID 재분배와 길이 축소 |
| [The Pitfall of Scaling Up](https://arxiv.org/abs/2606.21911) (2026) | 미확인 | KDD | 3 | 트랜스포머 추천 스케일업 시 인기 편향 증폭 |
| [More Than What Was Chosen](https://openreview.net/forum?id=WYfDoB44xy) (2026) | KAIST, SK Telecom | ICLR 2026 | 2 | 노이즈 선호를 넘는 LLM 설명 가능 추천 |
| [MuonRec](https://arxiv.org/abs/2603.00416) (2026) | Kuaishou 산학 (메모 기준) | arXiv | 2 | 생성형 추천에 Muon 옵티마이저 적용 |
| [Rethinking Semantic–Collaborative Integration](https://arxiv.org/abs/2604.22195) (2026) | 미확인 | SIGIR | 0 | LLM 시맨틱 임베딩과 협업 임베딩의 정렬 대신 보완 융합 |
| [FAIR](https://arxiv.org/abs/2512.11254) (2025) | Huawei Noah's Ark | arXiv | 0 | 생성형 추천의 어텐션 노이즈 완화 |
| APAO (2026) | Tsinghua (메모 기준) | 미확인 | 미확인 | 생성형 추천의 학습·추론 불일치 해소 |

## 4. LLM을 보조 신호로 쓰는 논문 (Enhancer)

기존 랭킹 모델을 유지하면서 도입하는 경로라서, 첫 단계로 현실적인 선택지입니다.

| 논문 | 기관 | 게재처 | 인용 | 내용 |
|---|---|---|---|---|
| [KAR](https://arxiv.org/abs/2306.10933) (2023) | Huawei, SJTU | RecSys | 241 | LLM의 세계 지식을 미리 계산해 CTR 모델에 투입 |
| [LLM-ESR](https://arxiv.org/abs/2405.20646) (2024) | CityU HK | NeurIPS | 184 | LLM 임베딩으로 롱테일 시퀀셜 추천 보강 |
| [LLMEmb](https://arxiv.org/abs/2409.19925) (2024) | CityU HK | AAAI | 105 | LLM을 시퀀셜 추천용 임베딩 생성기로 사용 |
| [ClickPrompt](https://arxiv.org/abs/2310.09234) (2023) | SJTU, Huawei | WWW | 77 | CTR 모델이 언어 모델용 프롬프트 생성 |
| [BAHE](https://arxiv.org/abs/2403.19347) (2024) | Ant Group | SIGIR | 67 | 긴 텍스트 행동 이력의 LLM 기반 CTR 예측 |
| [CTRL](https://arxiv.org/abs/2306.02841) (2023) | Huawei | ACM TORS | 57 | 협업 모델과 언어 모델 정렬 |
| [Uni-CTR](https://arxiv.org/abs/2312.10743) (2023) | Huawei | ACM TOIS | 47 | LLM 기반 멀티 도메인 CTR 통합 |
| [FLIP](https://arxiv.org/abs/2310.19453) (2023) | SJTU, Huawei | RecSys | 36 | ID 모델과 언어 모델의 세밀한 정렬 |

노션 서베이의 저자는 이 범주에서 KAR을 가장 주목할 연구로 꼽습니다. LLM을 추천기가 아니라 세계 지식의 공급원으로 한정하고 출력을 미리 계산해 서빙 비용 문제를 피했다는 설계 판단 때문입니다.

## 5. 서베이와 분석

| 논문 | 기관 | 게재처 | 인용 |
|---|---|---|---|
| [A Survey on LLMs for Recommendation](https://arxiv.org/abs/2305.19860) | USTC 외 | World Wide Web 저널 | 1,096 |
| [Recommender Systems in the Era of LLMs](https://arxiv.org/abs/2307.02046) | HK PolyU 외 | IEEE TKDE | 634 |
| [How Can Recommender Systems Benefit from LLMs](https://arxiv.org/abs/2306.05817) | SJTU, Huawei | ACM TOIS | 485 |
| [Cold-Start Recommendation in the Era of LLMs](https://arxiv.org/abs/2501.01945) | 다기관 | arXiv | 77 |
| [Scaling New Frontiers](https://arxiv.org/abs/2412.00714) | Huawei Noah's Ark | arXiv | 41 |
| [A Survey on Generative Recommendation](https://arxiv.org/abs/2510.27157) | 미확인 | AI Open (저널) | 38 |

## 6. LLM as Agent / Simulator / 설명

인용수와 게재처 기준은 잘 충족하지만 광고추천 랭킹 백본과는 거리가 있습니다. 대화형 추천이나 오프라인 평가용 시뮬레이션을 검토할 때의 참고 목록입니다.

| 논문 | 기관 | 게재처 | 인용 | 내용 |
|---|---|---|---|---|
| [LLMs as Zero-Shot Conversational Recommenders](https://arxiv.org/abs/2308.10053) (2023) | UCSD, Netflix | CIKM | 322 | LLM은 콘텐츠 지식에 강하고 협업 지식에 약하다는 진단 |
| [InteRecAgent](https://arxiv.org/abs/2308.16505) (2023) | Microsoft Research | ACM TOIS | 317 | LLM이 추천 도구를 호출하는 대화형 에이전트 |
| [RecMind](https://arxiv.org/abs/2308.14296) (2023) | Amazon | NAACL | 265 | LLM 기반 추천 에이전트 |
| [Agent4Rec](https://arxiv.org/abs/2310.10108) (2023) | NUS | SIGIR | 216 | LLM 에이전트로 사용자 시뮬레이션 |
| [XRec](https://arxiv.org/abs/2406.02377) (2024) | HKU | EMNLP | 126 | LLM 기반 추천 설명 생성 |
| [How Reliable is Your Simulator?](https://arxiv.org/abs/2403.16416) (2024) | 미확인 | WWW | 57 | LLM 사용자 시뮬레이터의 정답 누설 등 한계 분석 |

## 7. 평가와 근거의 신뢰도

LLM 추천 논문의 성능 주장을 어디까지 믿을지 판단하는 근거입니다.

| 논문 | 기관 | 게재처 | 인용 | 내용 |
|---|---|---|---|---|
| [Biases in ChatGPT-based Recommender Systems](https://arxiv.org/abs/2401.10545) (2024) | Politecnico di Bari | ACM TORS | 95 | 공급자 공정성, 시간 안정성, 최신성 편향 |
| [LLMs as RecSys: Popularity Bias](https://arxiv.org/abs/2406.01285) (2024) | 미확인 | arXiv | 47 | LLM 추천의 인기 편향 분석 |
| [Do LLMs Memorize Recommendation Datasets?](https://arxiv.org/abs/2505.10212) (2025) | Politecnico di Bari | SIGIR | 40 | LLM의 MovieLens-1M 암기 여부 검증 |

---

## 우선 읽을 7편

| 순서 | 논문 | 이유 |
|---|---|---|
| 1 | TOIS 서베이 | 전체 지형과 "어디에, 어떻게 LLM을 넣는가" 분류를 잡는 용도 |
| 2 | TIGER | 이후 생성형 추천 대부분이 전제로 삼는 Semantic ID 개념 |
| 3 | HSTU | 빅테크 + ICML + 높은 인용수로 세 기준을 모두 충족 |
| 4 | OneRec | 미게재 arXiv지만 2025년 논문 중 인용수가 압도적 |
| 5 | GPR | 광고 도메인에서 단일 생성 모델 전면 배포 사례 |
| 6 | LLaTTE + ExFM | Meta 광고의 스케일링 법칙과 서빙 구조, 인프라 비용 판단에 직접 사용 |
| 7 | KAR | 기존 CTR 모델을 유지하는 저위험 경로의 대표작 |

## 유의사항

- 학술지(저널) 게재는 서베이와 InstructRec, CoLLM, BIGRec, CTRL, Uni-CTR, InteRecAgent, LIGER 정도이고, 이 분야의 주요 성과는 대부분 학회(KDD, RecSys, WWW, NeurIPS, ICML) 논문입니다. "학술지 게재"만 기준으로 삼으면 핵심 산업 논문이 거의 빠집니다.
- COBRA와 PLUM의 게재처는 Semantic Scholar 표기이며 원문 확인이 필요합니다.
- ICLR 2026 논문 5편은 Semantic Scholar가 arXiv 공개 연도(2025)로 표기합니다. 게재처는 위키 스터디 페이지 기준입니다.
- NEO와 GLIDE는 위키 제목과 arXiv 제목이 달라서, 같은 날짜 스터디 글의 Spotify 논문 두 편에 대응시켰습니다. 링크가 다르면 수정이 필요합니다.
- 아지트 스터디 글에서 논문 제목을 자동 추출했기 때문에, 영어 4단어 미만이거나 숫자로 시작하는 제목은 누락됐을 수 있습니다.
- 인용이 거의 없거나 소속을 확인하지 못해 뺀 논문: [AdNanny](https://arxiv.org/abs/2602.01563), [Fine-Tuned LLM as a Complementary Predictor](https://arxiv.org/abs/2605.27856), MiniOneRec, KuaiFormer, OneSearch-V2, TM20K, TGR, Benchmark Leakage Trap, The Recall Ceiling of LLM Recommendation Reranking 등.

---

## 개별 검토

### GenRec (Netflix, 2026)

[GenRec: An LLM-Backed Recommendation Ranker at Netflix](https://arxiv.org/abs/2608.10257) — 2026년 8월 공개, 9쪽 프리프린트.

**세 기준 평가**

| 기준 | 평가 |
|---|---|
| 출처 기관 | Netflix (저자 12명 중 11명) — 충족 |
| 게재 여부 | arXiv만 있고 학회·저널 표기 없음 — 미충족 |
| 인용수 | 미확인 (공개 3개월 시점) |

**내용**

- **대상**: Netflix 타이틀 전체 카탈로그 랭킹. 광고, 입찰, CTR/CVR 언급 없음.
- **구조**: 오픈소스 LLM을 Netflix 데이터로 적응시킨 사내 파운데이션 LLM(1단계)을 랭킹 라벨과 리워드로 후학습(2단계). 모델 크기는 약 1B, 약 10B. 기반 모델 이름은 비공개.
- **입력**: 시청 이력과 메타데이터를 자연어로 풀어 쓴 텍스트. Semantic ID는 쓰지 않고, LLM의 풀링된 hidden state와 아이템별 학습 임베딩을 결합해 카탈로그 전체에 softmax.
- **서빙**: vLLM 기반 prefill-only 구성으로 forward 1회에 전체 후보 순위 산출. 배치 계산 지면 대상.

**보고된 성과**

- 오프라인: 프로덕션 모델보다 2단계 라벨을 약 40배 적게 쓰고 MRR 상대 +1.6%
- 온라인: 트래픽 약 10%, 4주 A/B에서 핵심 지표 상대 +0.006%
- 비용: 컨텍스트를 약 5,000토큰에서 약 1,700토큰으로 줄여 서빙 비용 약 1/3

**광고추천 도입 관점의 한계**

- 온라인 이득이 매우 작고 신뢰구간·검정 방법이 없음
- 지연시간 수치가 없고, 배치 지면이라 가능한 구조라서 실시간 광고 랭킹에는 그대로 옮기기 어려움
- 베이스라인이 "수년간 튜닝한 판별 모델"로만 기술되고 PLUM·OneRec·TIGER와 비교 없음
- 데이터·모델·코드 모두 비공개

**가져갈 부분**

- 라벨 효율: 전환처럼 라벨이 희소한 광고 과제에 시사점
- 2단계 학습의 기여도: 범용 LLM 대비 1단계 적응이 MRR +10~20%, 2단계 후학습이 추가 +35~50%. 범용 LLM을 도메인 적응 없이 바로 쓰면 안 된다는 근거
- 컨텍스트 엔지니어링: 저신호 이벤트 제거, 반복 시청 압축, 오래된 이력 요약
