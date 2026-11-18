# 광고 어뷰징 탐지 모델 레퍼런스 리서치

- 작성일: 2026-11-18
- 범위: 공개 자료만 조사 (사내 위키 제외)
- 서지 표기 원칙
  - 웹 검색으로 제목·저자·발표처를 확인한 것만 인용
  - 확인하지 못한 항목은 "미확인"으로 표시
  - 벤더 자료의 수치는 자체 보고이므로 근거로 쓰지 않음

---

## 1. 요약과 권장 접근

### 1.1 결론

- 1차로 시도할 방식은 두 가지
  - (A) 키 단위 스코어 모델: 집계·그래프 피처 + GBDT + weak label
  - (B) 그래프 집단 탐지: 공동 방문·공동 입찰 그래프에서 밀집 군집 추출
- 두 방식은 대체가 아니라 보완 관계
  - (A)는 개별 키의 이상 행동을 잡음
  - (B)는 개별로는 정상처럼 보이는 집단 어뷰징을 잡음
  - (B)의 결과는 (A)의 피처와 라벨 소스로 다시 들어감
- 룰은 없애지 않고 층으로 남김
  - 공개된 업계 구조는 모두 "룰/리스트 → ML → 수동 검수"의 다층 구조 (Google, Criteo, Microsoft, Amazon)
  - GIVT(데이터센터 IP, 알려진 봇)는 리스트·룰이 더 싸고 설명 가능함

### 1.2 (A) 키 단위 스코어 모델

- 정의
  - 키(adid/aid/sdkId, IP, slot, cTag)마다 어뷰징 확률을 내는 지도 분류 모델
- 선택 이유
  - 서빙 구조와 단위가 같음. camp-api가 이미 키 단위 필터와 3단 액션을 지원
  - 테이블형 집계 피처에서는 GBDT가 표준. TalkingData 대회, FDMA 2012 모두 트리 앙상블이 상위
  - `daq_ft_ana_*` 집계 테이블을 그대로 피처 원천으로 쓸 수 있음
- 라벨
  - 음성(정상) 프록시: 전환·구매 등 사람만 하는 행동이 있는 키 (Amazon SLIDR, Botcha 방식)
  - 양성(어뷰징) 프록시: 기존 룰 필터 적중 + 고정밀 휴리스틱
  - 나머지는 unlabeled로 두고 PU 또는 가중 학습
- 핵심 위험
  - 룰 라벨로 학습하면 모델이 룰을 복제함
  - 대응: 룰 입력 변수를 피처에서 빼거나, 룰 하나를 홀드아웃해 일반화 측정 (5.1 참고)

### 1.3 (B) 그래프 집단 탐지

- 정의
  - 유저–매체, IP–매체 이분 그래프에서 비정상적으로 겹치는 군집을 찾는 비지도 방법
- 선택 이유
  - 라벨이 필요 없음
  - 자산이 이미 있음: `daq_graph_cobid_media_ip_*`, `daq_gnn_emb_*`
  - 디스플레이 광고에서 검증된 선례가 있음: m6d의 co-visitation network (KDD 2013), Zemanta 블로그, 카카오 if(kakao) 2024 발표
- 한계
  - 군집이 나와도 "어뷰징인지"는 별도 검증이 필요 (전환 품질, 수동 검수)
  - 공용 IP(통신사 NAT, 회사망)가 가짜 밀집을 만듦. 사전 제거 필요

### 1.4 지금 하지 않는 것

- end-to-end GNN 분류, 시퀀스 딥러닝
  - 이유: 신뢰할 라벨이 먼저 필요. 라벨 품질이 병목인 상태에서 모델 복잡도를 올려도 평가를 못 함
  - 시점: (A)의 평가 체계가 선 뒤 2차로 검토

---

## 2. 질문별 정리

### 2.1 문제 정의: 어뷰징 유형과 로그 신호

#### 업계 분류

- IVT (Invalid Traffic)
  - 정의: 진짜 사용자 관심에서 나오지 않은 광고 트래픽. 악의 여부와 무관
  - 출처: MRC "Invalid Traffic Detection and Filtration Guidelines" (2015 제정, 이후 개정)
- GIVT (General IVT)
  - 정의: 리스트·표준 파라미터 검사로 걸러지는 트래픽
  - 예시: 알려진 크롤러, 데이터센터 IP, 비브라우저 User-Agent, prefetch
  - 우리 대응: 룰·블록리스트 층
- SIVT (Sophisticated IVT)
  - 정의: 사람을 흉내 내서 고급 분석이 필요한 트래픽
  - 예시: 봇넷, 탈취 기기, 도메인·앱 스푸핑, 위치 위조, 인센티브 조작
  - 우리 대응: 모델 스코어 층
- TAG Certified Against Fraud
  - 정의: MRC 기준에 따른 GIVT/SIVT 필터링과 ads.txt, sellers.json 채택을 요구하는 인증
  - 시사점: ads.txt/sellers.json은 "판매 권한"만 증명. 사람이 봤는지는 증명 못 함
- 주의
  - MRC 원문 PDF는 직접 열람하지 못함. 위 정의는 2차 자료(Pixalate, PR Newswire) 기준

#### 유형별 로그 신호

| 유형 | 정의 | 로그에서 보이는 신호 | 주 탐지 단위 |
|---|---|---|---|
| 봇·데이터센터 트래픽 | 자동화 프로그램이 만든 요청 | 호스팅 ASN IP, UA 이상, 24시간 균일 분포, 요청 간격 규칙성 | IP, user key |
| 클릭 사기 (click spam) | 관심 없는 클릭을 대량 생성 | CTR 과다, 클릭 후 전환 0, 노출–클릭 간격 과소, 중복 클릭 | slot, user key, IP |
| 클릭 인젝션 | 설치 직전에 가짜 클릭을 끼워 넣어 기여 탈취 (Android) | CTIT(클릭–설치 시간) 극단적으로 짧음 | cTag(channel), slot |
| 클릭 스패밍·플러딩 | 대량 클릭을 뿌려 오가닉 설치 기여 탈취 | CTIT 긴 꼬리, 클릭 대비 전환율 극저 | cTag, slot |
| 노출 사기 | 보이지 않는 노출 생성 (숨김 프레임, 백그라운드, pay-per-view) | bid·imp는 많고 viewable imp·click 없음, 한 세션에 과다 노출 | slot, 매체 |
| 매체(지면) 사기 | 매체가 자기 지면 트래픽을 부풀림 | 매체 간 유저 과다 중복, 유저당 수익 이상, 콘텐츠 무관 매체끼리 군집 | slot, 매체 |
| 도메인·앱 스푸핑 | 비싼 매체로 위장 | 선언 매체와 실제 신호 불일치, ads.txt 불일치 | 매체, cTag |
| 설치·전환 사기 | 디바이스 팜, 에뮬레이터, SDK 스푸핑 | 신규 기기 비율 과다, adid 리셋 반복, 설치 후 이벤트 없음 | user key, IP |
| 인센티브 트래픽 | 보상을 받고 수행한 클릭·설치 | 설치 직후 이탈, 리텐션·후행 이벤트 급락, 캠페인 시간대 집중 | cTag, slot |
| 크라우드 사기 | 사람을 고용한 분산 공격 | 개별은 정상, 집단은 동기화 (같은 대상, 같은 시간대) | 집단 (그래프) |

- 근거 논문
  - 클릭 스팸 측정: Dave et al., SIGCOMM 2012
  - 노출 사기: Springborn & Barford, USENIX Security 2013
  - 광고 거래소 사기: Stone-Gross et al., IMC 2011
  - 봇넷 클릭 사기: Pearce et al., CCS 2014 (ZeroAccess)
  - 앱 내 클릭 사기: MAdFraud (MobiSys 2014)
  - 크라우드 사기: Tian et al., WWW 2015

### 2.2 피처

#### 원칙

- 단위를 서빙 필터 키와 맞춤
  - 설명: 스코어를 바로 필터로 바꿀 수 있어야 함
  - 예시: slot 단위 스코어 → slot 필터의 bid rate 축소
- 단일 통계보다 "기준 대비 편차"를 씀
  - 설명: 매체·앱마다 정상 CTR, CTIT가 다름
  - 예시: slot CTR 자체가 아니라 같은 매체 유형 내 z-score
  - 근거: AppsFlyer는 CTIT 하나로는 우회된다고 명시. 앱별 기준선 비교를 권장

#### 단위별 피처 후보

| 단위 | 피처 종류 | 예시 | 근거 |
|---|---|---|---|
| user key (adid/aid/sdkId) | 빈도·속도 카운터 | 1분/1시간/1일 창의 bid, imp, click 수 | SLIDR, TalkingData |
| user key | 다양성 | 고유 IP 수, 고유 slot 수, 고유 UA 수 | FDMA 2012 |
| user key | 시간 패턴 | 시간대 분포 엔트로피, 클릭 간격 분산, 다음 클릭까지 시간 | TalkingData, FDMA 2012 |
| user key | 후행 품질 | 전환 유무, 전환 후 이벤트 수 | SLIDR, Botcha |
| IP | 엔티티 카운터 | IP당 고유 user key 수, 세션 수 | SLIDR |
| IP | 네트워크 속성 | ASN 유형(호스팅/통신사), 프록시·VPN 여부, 국가 불일치 | Adjust Anonymous IP 필터, Amazon IP 논문 |
| IP | 임베딩 | 비지도 IP 임베딩 (autoencoder) | Amazon KDD 2023 워크숍 |
| slot·매체 | 비율 이상치 | CTR, CVR, click→conv 비율, viewable 비율 | Criteo 헬프센터 (activity ratio 룰) |
| slot·매체 | 수익성 | 유저당 수익 (ROI 프록시) | ViceROI, CCS 2013 |
| slot·매체 | 유저 구성 | 신규 user key 비율, 상위 유저 집중도(top-k 비중) | AppsFlyer (신규 기기 비율) |
| cTag (cp, channel) | 기여 시간 분포 | CTIT 분위수, 0~10초 비중, 긴 꼬리 비중 | Adjust Distribution Modeling, AppsFlyer |
| cTag | 클릭 효율 | 클릭당 전환율, click-to-click 간격 | AppsFlyer |
| 그래프 | 공동 방문 | 매체 A 유저 중 매체 B도 방문한 비율, 군집 계수 | m6d KDD 2013, Zemanta |
| 그래프 | 밀집도 | 속한 밀집 부분그래프의 의심도, 군집 크기 | FRAUDAR, CopyCatch |
| 그래프 | 구조 지표 | degree, personalized PageRank, 연결요소 크기 | 카카오 if(kakao) 2024 |
| 그래프 | 임베딩 | GNN·heterogeneous graph 임베딩 | GEM |
| 시퀀스 | 행동 순서 | 이벤트 유형·간격 시퀀스 | MCCF (Alibaba) |

- 피처 공학의 실증 결과
  - FDMA 2012 결론: 세밀한 시간 단위의 시계열 피처가 탐지에 결정적
  - TalkingData 상위 솔루션: 시간 창별 count, unique count, 누적 count, 다음 클릭까지 시간 (1위 원문은 미확인, 2차 요약 기준)
  - SLIDR: 시각을 단위원에 매핑. 봇은 일·주 주기성이 약함

### 2.3 라벨

#### 결론

- 완전한 정답은 없음. 업계는 "프록시 라벨 여러 개 + 소량의 고정밀 수동 라벨" 조합을 씀
- 정상(human) 쪽 프록시가 어뷰징 쪽 프록시보다 믿을 만함
  - 이유: 봇은 계속 바뀌어 "나쁜 신호"를 고정하기 어려움. 구매·전환은 위조 비용이 큼 (Botcha 논문의 주장)

#### 라벨 소스

| 소스 | 정의 | 장점 | 단점 | 근거 |
|---|---|---|---|---|
| 룰 적중 | 기존 필터가 잡은 키를 양성으로 | 즉시 사용 가능, 대량 | 룰 복제 위험, 룰이 못 잡는 유형은 음성으로 오염 | Criteo, Microsoft |
| high-hurdle 행동 | 구매·전환·로그인 등 사람만 하는 행동을 정상으로 | 위조 비용이 큼 | 전환 사기에는 취약, 정상 중 일부만 포착 | SLIDR, Botcha |
| 후행 품질 역추적 | 설치 후 리텐션·이벤트가 없는 소스를 의심 | 광고주 가치와 직결 | 지연 (수일~수주), 소스 단위로만 가능 | AppsFlyer post-attribution |
| 허니팟·블러프 광고 | 사람이 누를 이유가 없는 미끼 광고 | 독립적 정답에 가까움 | 인벤토리 비용, 운영 부담 | Haddadi 2010, Dave 2012 |
| 광고주 클레임·환불 | 신고·환불 이력 | 실제 피해 기준 | 희소, 편향 (큰 광고주 위주) | Google (수동 리뷰 입력) |
| 수동 검수 | 전문가가 표본을 판정 | 고정밀 | 소량, 비쌈 | Sculley KDD 2011, DEC |
| 외부 리스트 | IAB/TAG 봇·데이터센터 리스트 | 표준 | GIVT만 포착 | Criteo, MRC |
| 그래프 군집 | 밀집 군집 소속을 양성 후보로 | 룰과 독립적 | 군집=어뷰징 검증 필요 | m6d, 카카오 |

#### 불완전 라벨 학습 방법

- PU learning
  - 정의: 양성 일부와 미라벨만으로 분류기를 학습
  - 설명: 미라벨을 음성으로 두고 학습한 뒤 라벨 빈도로 보정. 가정은 "라벨된 양성이 양성 전체의 무작위 표본"
  - 주의: 이 가정은 우리 상황에서 깨짐. 룰이 잡는 어뷰징은 특정 유형에 치우침. Botcha가 이 편향을 다룸
  - 예시: 전환 있는 키 = 양성(human), 나머지 = unlabeled
- Weak supervision (라벨링 함수 결합)
  - 정의: 여러 개의 불완전한 라벨링 함수를 결합해 확률 라벨을 만듦
  - 설명: 룰 각각을 라벨링 함수로 보고 정확도·상관을 추정해 합침
  - 예시: "CTIT<10초 비중>15%", "IP당 adid>N", "전환 있음(정상)" 등을 각각 투표로
  - 근거: Snorkel DryBell (Google, SIGMOD 2019)
- 다단계 학습 (저정밀 대량 → 고정밀 소량)
  - 정의: 자동 라벨로 표현을 먼저 학습하고 수동 라벨로 미세조정
  - 근거: Meta DEC (USENIX Security 2021). 어뷰징 유형별 멀티태스크로 1단계, 사람 라벨로 2단계
- 노이즈 라벨 정제
  - 정의: 모델 예측과 라벨이 크게 어긋나는 샘플을 찾아 제거·재검수
  - 용도: 수동 검수 대상 선정에 유용 (검수 예산을 불일치 샘플에 집중)

### 2.4 모델과 학습

| 방식 | 전제 | 장점 | 단점 | 필요한 데이터 | 대표 레퍼런스 |
|---|---|---|---|---|---|
| 지도 (GBDT) | 쓸 만한 라벨이 있음 | 테이블 피처에 강함, 중요도 해석 가능, 배치 스코어링 쉬움 | 라벨 편향을 그대로 학습 | 키 단위 집계 + weak label | FDMA 2012, TalkingData |
| 지도 (DNN) | 대량 데이터, 실시간 추론 | 임베딩·교차 학습, 슬라이스 보정과 결합 | 운영 복잡도 | 이벤트 단위 피처 + 프록시 라벨 | SLIDR (AAAI 2023) |
| 비지도 이상탐지 | 어뷰징은 드물고 다르다 | 라벨 불필요, 새 유형 발견 | 이상≠어뷰징, 오탐 많음 | 키 단위 집계 | Amazon IP autoencoder |
| 수익성 이상탐지 | 사기는 높은 ROI가 필요 | 튜닝 파라미터 거의 없음, 공격 유형 무관 | 매체 단위만, 검수 후보 선정용 | 매체별 유저당 수익 | ViceROI (CCS 2013) |
| PU·준지도 | 한쪽 클래스만 신뢰 가능 | 우리 라벨 상황과 일치 | 표본 편향 가정에 민감 | 정상 프록시 + 미라벨 | Botcha |
| 밀집 부분그래프 | 어뷰징은 자원을 공유해 밀집 블록을 만듦 | 라벨 불필요, 위장에 대한 이론적 한계 제공(FRAUDAR) | 저밀도·분산 공격은 놓침 | 이분 그래프 (유저–매체, IP–매체) | FRAUDAR, CopyCatch, CatchSync, HoloScope |
| 동기화 군집 | 같은 대상에 비슷한 시각에 행동 | 느슨한 동기화까지 포착 | 실시간 아님, 임계값 설계 필요 | (키, 대상, 시각) 로그 | SynchroTrap, Crowd Fraud (WWW 2015) |
| 공동 방문 그래프 | 정상 매체끼리는 유저가 과하게 겹치지 않음 | 디스플레이 RTB에서 검증됨, 구현 단순 | 유사 콘텐츠 매체의 자연 중복과 구분 필요 | 유저–매체 방문 로그 | m6d (KDD 2013), Zemanta, 카카오 |
| GNN | 이웃 정보가 유용하고 라벨이 있음 | 집단 신호를 피처로 자동 학습 | 라벨 필요, 위장 이웃에 취약, 운영 비용 | 이종 그래프 + 노드 라벨 | GEM (CIKM 2018) |
| 시퀀스·멀티모달 | 이벤트 순서에 정보가 있음 | 사람 흉내 공격에 대응 | 데이터·연산 비용 큼 | 유저별 이벤트 시퀀스 | MCCF (Alibaba) |

- 방식 선택의 근거
  - 밀집 부분그래프의 직관
    - 정의: 어뷰저는 기기·IP·계정을 재사용하므로 그래프에 비정상적으로 촘촘한 블록이 생김
    - 예시: 100개 adid가 같은 20개 slot에서만 같은 시간대에 클릭
    - 한계: 어뷰저가 인기 대상에 간선을 추가해 위장. FRAUDAR는 이 위장에 견디도록 설계
  - GEM의 관찰 (Alipay)
    - 악성 계정의 두 약점: device aggregation(소수 기기에 몰림), activity aggregation(시간적으로 몰림)
    - 시사점: 우리도 IP·기기 공유와 시간 집중을 그래프 간선과 피처로 표현 가능
  - 크라우드 사기의 세 특성 (Baidu)
    - moderateness(개별은 적당), synchronicity(시간 동기), dispersivity(대상 분산)
    - 시사점: 키 단위 임계 룰로는 구조적으로 못 잡는 유형. 집단 단위 탐지가 필요

### 2.5 평가와 운영

#### 오프라인 평가 (라벨이 불완전할 때)

- 정확도·AUC를 그대로 믿지 않음
  - 이유: 라벨이 프록시라서 "프록시 재현력"만 측정됨
- 프록시 지표 조합 (Amazon SLIDR의 방식)
  - IVR (Invalidation Rate)
    - 정의: 전체 중 무효로 판정한 비율
    - 용도: recall의 대리. 단독으로는 오해 소지
  - FPR 프록시
    - 정의: 전환(구매) 클릭 중 무효로 판정한 비율
    - 가정: 전환 클릭은 사람이고 정상 트래픽을 대표
  - Robotic coverage
    - 정의: 고정밀 휴리스틱(예: 시간당 클릭 k회 초과 세션)이 잡은 것 중 모델이 잡은 비율
    - 용도: recall의 더 정확한 대리
- 룰 일반화 검증
  - 방법: 룰 하나의 적중을 학습 라벨에서 빼고, 모델이 그 룰 적중을 얼마나 복원하는지 측정
  - 의미: 룰을 외우지 않고 어뷰징 패턴을 배웠는지 확인
- 시간 분할
  - 방법: 학습 기간 이후 날짜로 검증 (TalkingData 상위권의 day 8 학습/day 9 검증)
  - 이유: 무작위 분할은 같은 키가 양쪽에 들어가 과대평가
- 검수 표본 평가
  - 방법: 스코어 구간별 층화 표본을 수동 검수해 구간별 정밀도 추정
  - 근거: Sculley KDD 2011 (층화 표본, 전문가 검수)

#### 온라인 검증과 단계적 차단

- 기존 액션 구조에 스코어 구간을 매핑
  - 낮은 의심: 관찰만 (shadow 스코어 로깅)
  - 중간 의심: bid rate 축소
  - 높은 의심: MAX_ECPM_CAP
  - 매우 높은 의심 + 룰 동의: NO_BIDDING
  - 이 매핑은 조사 근거가 아니라 제안. Google도 "의심되지만 확신 없는 트래픽은 모니터링"하는 단계를 둠
- 버킷 A/B에서 볼 지표
  - 광고주 측: 필터 적용 버킷의 CVR, 전환당 비용, 후행 이벤트율 (오르면 걸러낸 것이 저품질)
  - 플랫폼 측: 매출 손실, 필터된 트래픽의 전환 비중 (높으면 오탐)
- bid rate 축소의 부가 가치
  - 설명: 100% 차단하지 않고 일부를 통과시키면 그 키의 후행 지표를 계속 관찰 가능
  - 용도: 차단된 키의 반사실 데이터 확보. 재학습 라벨 편향 완화

#### 오탐 비용

- 비대칭
  - 오탐: 정상 매체·유저의 매출 손실, 매체 이탈
  - 미탐: 광고주 예산 낭비, 전환 모델(pCTR/pCVR) 학습 데이터 오염
- 슬라이스별 보정
  - 정의: 전체 FPR 예산을 고정하고 슬라이스별 임계값을 따로 정함
  - 설명: 단일 임계값은 슬라이스마다 과소·과대 판정을 만듦 (SLIDR: 데스크톱 과소, 모바일 과대)
  - 예시: 매체 유형·OS·로그인 여부별 임계값. SLIDR은 이를 볼록 최적화로 풂

#### 적대적 적응과 재학습

- 드리프트는 전제
  - 근거: Microsoft adCenter는 5년간의 "적대적" 운영 경험을 정리. AppsFlyer는 CTIT 탐지를 우회하는 변종을 보고
- 재학습 주기
  - SLIDR은 매일 재학습·재보정
  - 우리는 배치가 일 단위이므로 일~주 단위가 현실적
- 가드레일 (SLIDR)
  - 입력: 시간대별 정상 라벨 밀도 하한. 라벨 유실 시 대량 무효화 방지
  - 모델: 지표가 튀는 모델의 배포 차단
  - 복구: 직전 안정 모델로 롤백, 트래픽 재처리 도구
- 피드백 루프
  - 문제: 차단한 키는 이후 데이터가 없어 라벨이 고정됨
  - 대응: 일부 통과(bid rate), 주기적 재심사(TTL)

#### 스코어 → 정책

- 임계값 결정
  - IVR–FPR 곡선의 무릎점 (SLIDR)
  - 또는 FPR 예산을 먼저 정하고 역산
- 검수 큐
  - ViceROI는 차단이 아니라 "검수할 후보를 줄이는" 용도로 설계
  - 중간 구간은 자동 조치보다 검수 큐가 안전

#### 설명 가능성

- 필요한 이유
  - 매체·광고주 문의 대응, 오탐 디버깅, 수동 검수 효율
- 방법
  - GBDT + 피처 기여도 (SHAP 등. 본 조사에서 별도 레퍼런스 확인은 안 함)
  - 그래프 방식은 "어떤 키들과 무엇을 공유했는지"가 그대로 근거
  - 스코어와 함께 상위 기여 피처 3개를 필터 메타데이터로 저장

### 2.6 업계 사례

- Google
  - 구조: 실시간 필터 + 오프라인 분석 + 수동 리뷰. 탐지 지연분은 사후 크레딧 (공식 헬프 페이지)
  - Tuzhilin 보고서 (2006): 당시 필터 대부분이 단순했고, 전환 데이터 활용과 데이터 마이닝을 권고
  - Sculley et al. (KDD 2011): 악성 광고 탐지. 자동·반자동 다층 구조, 전문가 라벨, 층화 표본 평가
  - Snorkel DryBell (SIGMOD 2019): 조직 내 기존 자원을 약한 라벨로 전환
  - 3ve 봇넷 (2018): White Ops와 공동 대응. "The Hunt for 3ve" 백서 원문은 미확인
- Meta
  - CopyCatch (WWW 2013): Page Like의 lockstep 행동 탐지
  - SynchroTrap (2014): 느슨한 동기화 군집. 한 달에 200만 개 이상 악성 계정 (2차 자료 기준. 발표처는 검색으로 재확인 못 함)
  - DEC (USENIX Security 2021): 이웃 집계 피처 2만 개 이상 + 다단계 멀티태스크 학습
- Amazon Ads
  - SLIDR (AAAI 2023): 2021년부터 운영. weak label, 슬라이스 보정, 가드레일
  - IP 탐지 (KDD 2023 워크숍): 비지도 IP 임베딩 + 프록시 라벨 지도 모델
- Microsoft
  - Kitts et al. (2015): adCenter 클릭 사기 탐지 5년. 시간당 약 10억 이벤트
  - ViceROI, DECAF, ZeroAccess 연구에 Microsoft 연구진·데이터 참여
- Criteo
  - 구조: 실시간 룰 엔진 + 오프라인 룰 엔진(ML) + IAB/TAG 리스트 + 외부 검증사(Pixalate, DoubleVerify)
  - 모델 구조를 다룬 논문·기술 블로그는 미확인
- Alibaba·Ant
  - MCCF (2021, arXiv/DeMaL@WWW): wide&deep 피처 + 행동 시퀀스 + 이종 그래프
  - GEM (CIKM 2018): 계정–기기 이종 그래프 GNN
- Baidu
  - Crowd Fraud Detection (WWW 2015, Tsinghua 공동)
- Tencent
  - 광고 어뷰징에 특화된 공개 논문은 미확인
  - WeChat click farming 탐지 논문(MHT-GNN, CMT)이 있으나 소속·발표처는 확인 못 함
- Twitter/X
  - 광고 사기에 특화된 공개 자료는 미확인
  - FRAUDAR가 Twitter 팔로워 그래프(14.7억 간선)로 검증됨
- m6d(Dstillery)·Zemanta
  - co-visitation network. DSP 입장의 매체 사기 탐지. 우리와 가장 비슷한 문제 설정
- 카카오
  - if(kakao) 2024 "그래프 기반 악성 유저군 탐지": 유저–매체 이분 그래프 → 매체–매체 투영 → SCC → 악성 커뮤니티 → 악성 유저
  - 공개 슬라이드에서 라벨·평가 방식은 확인 못 함 (텍스트 인코딩 깨짐)
- 어트리뷰션 업체
  - AppsFlyer Protect360: 실시간 차단 + post-attribution 탐지. CTIT, click-to-click, 신규 기기 비율, 앱별 기준선
  - Adjust: Click Injection Filter, Distribution Modeling(CTIT), Anonymous IP Filter, SDK Signature
- 검증 업체
  - HUMAN(White Ops): Methbot(2016), PARETO(2021) 등 봇넷 보고서. 탐지 규칙은 비공개
  - DoubleVerify: Fraud Lab 사례(ViperBot, CycloneBot, ShadowBot). 방법론 상세는 비공개
  - IAS: 방법론 공개 자료 미확인
- 공개 데이터셋·대회
  - FDMA 2012 (BuzzCity): 매체(publisher) 단위 사기 분류. 우리 slot 단위 문제와 유사
  - TalkingData AdTracking (Kaggle 2018): 클릭 → 앱 다운로드 예측. 약 1.8억 행, 양성 약 0.25%. 엄밀히는 사기 라벨이 아니라 전환 라벨
  - Avazu (Kaggle): CTR 예측 대회. 사기 라벨 없음 (검색 미확인, 일반 지식)

---

## 3. 레퍼런스 목록

### 3.0 등급과 핵심 명단

#### 결론

- 학술 논문 28건을 핵심 10 / 참고 18로 나눔
- 범용 방법론과 영향력·운영 근거가 모두 낮은 논문 11건은 문서에서 뺌
- 정독 대상은 핵심 10건
- 축은 세 개: 영향력(인용), 적합도(우리 문제와의 거리), 운영 근거(실서비스 배포·실데이터)
- 인용 수만으로 자르지 않음
  - 이유: 인용 상위는 범용 방법론이 차지함. 우리 문제에 가장 가까운 운영 사례(SLIDR, m6d)는 인용이 적음
- 소속(빅테크 여부)을 직접 가중하지 않고 운영 근거로 반영
  - 이유: 빅테크 논문의 가치는 회사 이름이 아니라 실트래픽 배포 경험에서 나옴
  - 예시: Snorkel DryBell은 Google 논문이지만 광고 어뷰징과 무관. m6d는 빅테크가 아니지만 DSP 운영 사례

#### 기준

- 영향력
  - 상(2점): 연평균 인용 20회 이상
  - 중(1점): 연평균 5~20회
  - 하(0점): 연평균 5회 미만
- 적합도
  - 상(2점): 광고 로그로 어뷰징을 직접 탐지하고, 키·매체 단위 필터로 옮길 수 있음
  - 중(1점): 광고 도메인이지만 관점이 다름(앱 분석, 측정), 또는 인접 도메인의 어뷰징 탐지
  - 하(0점): 일반 방법론
- 운영 근거
  - 상(2점): 자사 대규모 서비스에 배포했고 운영 결과를 보고함
  - 중(1점): 기업 실데이터로 검증함 (배포 여부는 확인 못 함 포함)
  - 하(0점): 공개 데이터, 시뮬레이션, 외부 측정
- 등급
  - 핵심: 합계 4점 이상이면서 적합도 중 이상
  - 제외: 적합도 하이면서 운영 근거 하 (범용 방법론), 또는 합계 1점 이하
  - 참고: 나머지
- 인용 수 출처
  - Semantic Scholar API, 2026-11-18 조회
  - 연평균 = 인용 수 ÷ (2026 − 발표 연도)
  - 주의: 최근 논문(2021 이후)은 인용이 덜 쌓여 불리함
- 운영 근거 판정의 한계
  - 초록·검색 요약 기준으로 판정함. 본문에서 배포 사실이 확인되면 "중"이 "상"으로 오를 수 있음 (CopyCatch, GEM, Crowd Fraud, MCCF)

#### 핵심 (10건)

| # | 논문 | 소속 | 연도 | 발표처 | 인용 | 연평균 | 영향력 | 적합도 | 운영 | 합계 | 읽는 목적 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | SynchroTrap | Facebook, Duke | 2014 | CCS | 297 | 24.8 | 상 | 중 | 상 | 5 | 느슨한 동기화 군집. Facebook·Instagram 배포 |
| 19 | Sculley et al. | Google | 2011 | KDD | 113 | 7.5 | 중 | 상 | 상 | 5 | 층화 표본·전문가 검수. 운영 시스템 |
| 24 | CopyCatch | Facebook, CMU | 2013 | WWW | 400 | 30.8 | 상 | 중 | 중 | 4 | 시간 조건부 lockstep 탐지 |
| 30 | GEM | Ant Financial | 2018 | CIKM | 448 | 56.0 | 상 | 중 | 중 | 4 | 기기·시간 집중 관찰, GNN 기준 |
| 34 | DEC | Facebook | 2021 | USENIX Security | 42 | 8.4 | 중 | 중 | 상 | 4 | 라벨 품질 격차를 흡수하는 다단계 학습. 2년 이상 운영 |
| 13 | FDMA 2012 정리 | SMU, BuzzCity | 2014 | JMLR | 99 | 8.3 | 중 | 상 | 중 | 4 | 매체 단위 분류, 시계열 피처 |
| 5 | ViceROI | UT Austin, MSR | 2013 | CCS | 80 | 6.2 | 중 | 상 | 중 | 4 | 수익성 기반 매체 탐지. 대형 광고 네트워크 데이터 |
| 29 | Crowd Fraud | Baidu, Tsinghua | 2015 | WWW | 67 | 6.1 | 중 | 상 | 중 | 4 | 키 단위 룰의 사각지대 |
| 20 | m6d co-visitation | m6d (Dstillery) | 2013 | KDD | 28 | 2.2 | 하 | 상 | 상 | 4 | 디스플레이 RTB의 같은 문제. DSP 배포 |
| 15 | SLIDR | Amazon | 2023 | AAAI (IAAI) | 4 | 1.3 | 하 | 상 | 상 | 4 | 라벨·평가·가드레일 운영. 2021년부터 배포 |

#### 참고 (18건)

| # | 논문 | 소속 | 연도 | 발표처 | 인용 | 연평균 | 영향력 | 적합도 | 운영 | 합계 |
|---|---|---|---|---|---|---|---|---|---|---|
| 33 | Snorkel DryBell | Google 외 | 2019 | SIGMOD | 161 | 23.0 | 상 | 하 | 상 | 4 |
| 23 | FRAUDAR | CMU | 2016 | KDD | 417 | 41.7 | 상 | 중 | 하 | 3 |
| 4 | Click-Spam 측정 | UT Austin, MSR | 2012 | SIGCOMM | 102 | 7.3 | 중 | 상 | 하 | 3 |
| 11 | DECAF | USC, MSR | 2014 | NSDI | 156 | 13.0 | 중 | 중 | 중 | 3 |
| 9 | Ad Injection at Scale | Google 외 | 2015 | IEEE S&P | 132 | 12.0 | 중 | 중 | 중 | 3 |
| 6 | Ad Exchange 사기 | UCSB | 2011 | IMC | 132 | 8.8 | 중 | 중 | 중 | 3 |
| 7 | ZeroAccess | UCSD, MSR 외 | 2014 | CCS | 73 | 6.1 | 중 | 중 | 중 | 3 |
| 17 | Kitts et al. | Microsoft | 2015 | Springer 챕터 | 26 | 2.4 | 하 | 중 | 상 | 3 |
| 32 | Botcha | 미확인 | 2020 | OHARS@RecSys | 7 | 1.4 | 하 | 상 | 중 | 3 |
| 18 | MCCF | Alibaba | 2021 | arXiv / DeMaL@WWW | 6 | 1.2 | 하 | 상 | 중 | 3 |
| 26 | CatchSync | Tsinghua, CMU | 2014 | KDD | 184 | 15.3 | 중 | 중 | 하 | 2 |
| 10 | MAdFraud | UC Davis | 2014 | MobiSys | 170 | 14.2 | 중 | 중 | 하 | 2 |
| 27 | HoloScope | CMU 외 | 2017 | CIKM | 93 | 10.3 | 중 | 중 | 하 | 2 |
| 31 | Bluff Ads | 미확인 | 2010 | SIGCOMM CCR | 135 | 8.4 | 중 | 중 | 하 | 2 |
| 28 | Detectives | UCSB | 2007 | WWW | 133 | 7.0 | 중 | 중 | 하 | 2 |
| 12 | Clicktok | Strathclyde | 2019 | WiSec | 47 | 6.7 | 중 | 중 | 하 | 2 |
| 8 | PPV 노출 사기 | UW–Madison 외 | 2013 | USENIX Security | 68 | 5.2 | 중 | 중 | 하 | 2 |
| 16 | Amazon IP 탐지 | Amazon | 2023 | KDD 워크숍 | 0 | 0.0 | 하 | 중 | 중 | 2 |

- 참고 안에서 먼저 볼 것
  - FRAUDAR: 운영 근거가 없어 기준상 참고지만, 접근 (B)를 구현할 때는 필독. 코드 공개
  - Click-Spam 측정: 외부 측정 연구라 참고지만, 유형 정리와 블러프 광고 방법론의 출발점
  - Kitts et al., Amazon IP 탐지: 빅테크 운영 사례. 본문을 열람하지 못했으므로 본문 확인 후 핵심 승격 여부 판단
  - Botcha: 정상 프록시 PU 설정이 우리와 같음. 실험 3 전에 읽을 것
- Snorkel DryBell이 합계 4점인데 참고인 이유
  - 적합도 하. Google 운영 사례지만 광고 어뷰징이 아니라 범용 라벨링

#### 표기 주의

- CatchSync(#26), HoloScope(#27)의 소속은 검색 요약에 나오지 않아 일반 지식으로 적었음. 인용 전 확인 필요

#### 등급 대상에서 뺀 것

- 표준·보고서·업계 문서·대회·발표 (#1, #2, #3, #14, #21, #22, #35~#41)
  - 이유: 인용 수로 비교할 수 없음
  - 예외로 챙길 것: #22 카카오 if(kakao) 2024 (사내 선행 작업), #14 TalkingData (피처 공학 실전 사례)

### 3.1 분류·표준

| # | 제목 | 저자/소속 | 연도 | 발표처 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|---|
| 1 | Invalid Traffic Detection and Filtration Guidelines | MRC | 2015~ | 가이드라인 | [보도자료](https://prnewswire.com/news-releases/media-rating-council-mrc-issues-final-version-of-invalid-traffic-detection-and-filtration-guidelines-addendum-300166843.html), [해설](https://www.pixalate.com/blog/mrc-definitions-sivt-givt-invalid-traffic-detection-filtration) | GIVT/SIVT 2단 분류 | 룰 층(GIVT)과 모델 층(SIVT) 분리의 근거. 원문 미열람 |
| 2 | Certified Against Fraud Guidelines | TAG | 2016~ | 인증 기준 | [tagtoday.net](https://www.tagtoday.net/fraud) | MRC 기준 필터링 + ads.txt 요구 | 공급망 검증은 스푸핑만 막음 |
| 3 | The Tuzhilin Report (Lane's Gifts v. Google) | A. Tuzhilin (NYU) | 2006 | 법원 제출 보고서 | [PDF](https://www.cis.upenn.edu/~mkearns/teaching/SponsoredSearch/Tuzhilin_Report.pdf) | Google 무효 클릭 탐지의 외부 감사 | 단순 필터 → 전환 데이터·데이터 마이닝 권고 |

### 3.2 측정 연구 (보안 학회)

| # | 제목 | 저자/소속 | 연도 | 발표처 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|---|
| 4 | Measuring and Fingerprinting Click-Spam in Ad Networks | Dave, Guha, Zhang (UT Austin, MSR) | 2012 | SIGCOMM | [PDF](https://www.cs.utexas.edu/~yzhang/papers/clickspam-sigc12.pdf) | 광고주가 클릭 스팸 비율을 독립 측정하는 방법. 블러프 광고 활용 | 허니팟으로 정답 근사 가능 |
| 5 | ViceROI: Catching Click-Spam in Search Ad Networks | Dave, Guha, Zhang | 2013 | CCS | [PDF](https://www.sysnet.ucsd.edu/sysnet/miscpapers/ccs13.pdf) | 유저당 수익이 비정상적으로 높은 매체를 탐지 | slot 단위 수익성 피처. 검수 후보 선정 |
| 6 | Understanding Fraudulent Activities in Online Ad Exchanges | Stone-Gross et al. (UCSB) | 2011 | IMC | [PDF](https://sites.cs.ucsb.edu/~vigna/publications/2011_IMC_clickfraud.pdf) | 광고 거래소 내부 시점의 사기와 봇넷 C&C 분석 | RTB 거래소 로그에서 보이는 신호 |
| 7 | Characterizing Large-Scale Click Fraud in ZeroAccess | Pearce et al. | 2014 | CCS | [MSR](https://microsoft.com/en-us/research/?p=167853) | 봇넷 클릭 사기 규모와 흐름 측정 | 봇넷은 다수 매체에 분산 |
| 8 | Impression Fraud in On-line Advertising via Pay-Per-View Networks | Springborn, Barford | 2013 | USENIX Security | [PDF](https://usenix.org/system/files/conference/usenixsecurity13/sec13-paper_springborn.pdf) | PPV 네트워크가 보이지 않는 노출을 생성 | 노출 사기 신호 (viewable 없는 imp) |
| 9 | Ad Injection at Scale | Thomas et al. (Google 외) | 2015 | IEEE S&P | [Google Research](https://research.google/pubs/pub43346) | 광고 인젝션 확장·바이너리 규모 측정 | 클라이언트 측 조작 유형 |
| 10 | MAdFraud: Investigating Ad Fraud in Android Applications | Crussell, Stevens, Chen (UC Davis) | 2014 | MobiSys | [PDF](https://www.cs.ucdavis.edu/~hchen/paper/mobisys2014.pdf) | 백그라운드 광고 요청·무상호작용 클릭 앱 탐지 | 앱(지면) 자체가 사기 주체 |
| 11 | DECAF: Detecting and Characterizing Ad Fraud in Mobile Apps | Liu, Nath, Govindan, Liu | 2014 | NSDI | [USENIX](https://www.usenix.org/conference/nsdi14/technical-sessions/presentation/liu_bin) | 광고 배치 사기 자동 탐지 | 지면 정책 위반 유형 |
| 12 | Clicktok: Click Fraud Detection using Traffic Analysis | Nagaraja, Shah (Strathclyde) | 2019 | WiSec | [arXiv](https://arxiv.org/pdf/1903.00733) | 클릭 타이밍 패턴 재사용 탐지, bait click | 시간 간격 피처, 미끼 기법 |

### 3.3 피처·지도 학습

| # | 제목 | 저자/소속 | 연도 | 발표처 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|---|
| 13 | Detecting Click Fraud in Online Advertising: A Data Mining Approach | Oentaryo et al. (SMU 외) | 2014 | JMLR 15 | [JMLR](https://jmlr.org/papers/v15/oentaryo14a.html) | FDMA 2012 대회 정리. 매체 단위 사기 분류 | 세밀한 시계열 피처 + 앙상블. slot 모델의 직접 선례 |
| 14 | TalkingData AdTracking Fraud Detection Challenge | TalkingData / Kaggle | 2018 | 대회 | [솔루션 정리 예](https://github.com/stys/kaggle-talkingdata-adtracking-fraud-detection) | ip·app·device·os·channel·시각만으로 전환 예측 | 창별 count/unique/next-click 피처, LightGBM, 시간 분할 검증. 1위 원문 미확인 |
| 15 | Real-Time Detection of Robotic Traffic in Online Advertising (SLIDR) | Muralidhar et al. (Amazon) | 2023 | AAAI (IAAI) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26844), [블로그](https://www.amazon.science/blog/invalidating-robotic-ad-clicks-in-real-time) | weak label DNN, 슬라이스별 임계값 최적화, 가드레일 | 라벨·평가·운영 전반의 가장 완결된 공개 사례 |
| 16 | Detecting Robotic and Compromised IPs in Digital Advertising | R R, Venkataramaiah, Som (Amazon) | 2023 | KDD 워크숍 (AI4Cyber) | [Amazon Science](https://www.amazon.science/publications/detecting-robotic-and-compromised-ips-in-digital-advertising) | 비지도 IP 임베딩 + 프록시 라벨 지도 모델 | IP 필터용 모델의 선례. "혼합 IP" 문제 |
| 17 | Click Fraud Detection: Adversarial Pattern Recognition over 5 Years at Microsoft | Kitts et al. (Microsoft) | 2015 | Real World Data Mining Applications (Springer) | [DOI](https://doi.org/10.1007/978-3-319-07812-0_10) | adCenter 탐지 시스템 설계와 5년 운영 결과 | 대규모 룰+모델 운영 경험. 본문 미열람 |
| 18 | Multimodal and Contrastive Learning for Click Fraud Detection (MCCF) | Li et al. (Alibaba) | 2021 | arXiv / DeMaL@WWW | [arXiv](https://arxiv.org/pdf/2105.03567) | 통계 피처 + 행동 시퀀스 + 이종 그래프 결합 | 2차 단계 모델 구조 참고 |
| 19 | Detecting Adversarial Advertisements in the Wild | Sculley et al. (Google) | 2011 | KDD | [Google Research](https://research.google/pubs/pub37195/) | 악성 광고 탐지. 다층 구조, 전문가 라벨, 층화 표본 | 불균형·적대적 환경의 평가·검수 설계 |

### 3.4 그래프 기반

| # | 제목 | 저자/소속 | 연도 | 발표처 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|---|
| 20 | Using Co-Visitation Networks for Detecting Large Scale Online Display Advertising Exchange Fraud | Stitelman et al. (m6d) | 2013 | KDD | [PDF](https://dstillery.com/wp-content/uploads/2016/07/Using-Co-Visitation-Networks.pdf) | 유저–매체 이분 그래프에서 매체 간 방문자 중복으로 사기 매체 탐지 | 우리 문제와 가장 가까움. 50% 중복 임계 |
| 21 | Detecting Suspicious Websites | A. Dimitriev (Zemanta) | 2017 | 기술 블로그 | [블로그](https://zemanta.github.io/2017/08/17/detecting-suspicious-websites) | co-visitation + 군집 계수로 의심 점수 | 구현이 단순. 유사 콘텐츠 중복 주의 |
| 22 | 그래프 기반 악성 유저군 탐지: 온라인 광고 도메인에서의 적용 | 카카오 광고추천팀 | 2024 | if(kakao) 2024 | [슬라이드](https://speakerdeck.com/kakao/ifkakao24-13) | 매체–매체 투영 그래프 → SCC → 악성 커뮤니티·유저 | 사내 기존 자산과 직결. 라벨·평가 내용은 미확인 |
| 23 | FRAUDAR: Bounding Graph Fraud in the Face of Camouflage | Hooi et al. (CMU) | 2016 | KDD (Best Paper) | [PDF](https://bhooi.github.io/papers/fraudar_kdd16.pdf) | 위장에 강한 밀집 부분그래프 탐지 | IP–매체 그래프에 바로 적용 가능. 코드 공개 |
| 24 | CopyCatch: Stopping Group Attacks by Spotting Lockstep Behavior in Social Networks | Beutel et al. (CMU, Facebook) | 2013 | WWW | 원문 링크 미확인 | 같은 대상에 같은 시간대에 행동하는 집단 탐지 | 시간 조건이 붙은 이분 블록. 저자 전체 명단 미확인 |
| 25 | SynchroTrap (Uncovering Large Groups of Active Malicious Accounts in Online Social Networks) | Cao et al. (Duke, Facebook) | 2014 | CCS (미확인) | 원문 링크 미확인 | 느슨한 동기화 행동의 유사도 군집 | 유사도 지표와 군집 알고리즘 분리 설계. 서지는 기억 기준 |
| 26 | CatchSync: Catching Synchronized Behavior in Large Directed Graphs | Jiang et al. | 2014 | KDD | [PDF](https://alexbeutel.com/papers/kdd2014.catchsync.pdf) | 연결 구조만으로 동기화·비정상 노드 탐지 | 피처 없이 구조만으로 가능 |
| 27 | HoloScope: Topology-and-Spike Aware Fraud Detection | Liu, Hooi, Faloutsos | 2017 | CIKM | [PDF](https://shenghua-liu.github.io/assets/pdf/papers/cikm2017-holoscope.pdf) | 구조 + 시간 급증을 결합한 의심도 | 저밀도 블록 대응 |
| 28 | Detectives: Detecting Coalition Hit Inflation Attacks in Advertising Networks Streams | Metwally, Agrawal, El Abbadi (UCSB) | 2007 | WWW | [PDF](https://www.www2007.cpsc.ucalgary.ca/papers/paper070.pdf) | 매체 연합이 공격 기기를 공유하는 사기 탐지 | 매체 쌍의 방문자 유사도. co-visitation의 원형 |
| 29 | Crowd Fraud Detection in Internet Advertising | Tian, Zhu (Tsinghua), Xia, Zhuang, Zhang (Baidu) | 2015 | WWW | [PDF](https://ml.cs.tsinghua.edu.cn/~jun/pub/crowd-fraud.pdf) | 사람 고용형 사기의 3특성과 군집 기반 탐지 | 키 단위 룰의 구조적 사각지대 |
| 30 | Heterogeneous Graph Neural Networks for Malicious Account Detection (GEM) | Liu et al. (Ant Financial) | 2018 | CIKM | [arXiv](https://arxiv.org/pdf/2002.12307) | 계정–기기 이종 그래프 GNN | device/activity aggregation 관찰 |

### 3.5 라벨·학습 방법론

| # | 제목 | 저자/소속 | 연도 | 발표처 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|---|
| 31 | Fighting Online Click-Fraud Using Bluff Ads | H. Haddadi | 2010 | ACM SIGCOMM CCR | [arXiv](https://www.arxiv.org/abs/1002.2353) | 사람이 누를 이유가 없는 미끼 광고 | 독립 라벨 소스. 임계 기반 탐지와 병용 권고 |
| 32 | Botcha: Detecting Malicious Non-Human Traffic in the Wild | Dhamnani et al. | 2020 | OHARS@RecSys | [arXiv](https://arxiv.org/abs/2103.01428) | 구매·CAPTCHA 통과를 양성(human)으로 둔 PU 학습 | 정상 프록시 + 표본 편향 보정. 소속 미확인 |
| 33 | Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale | Bach et al. (Google 외) | 2019 | SIGMOD (Industrial) | [arXiv](https://arxiv.org/pdf/1812.00417) | 조직 내 비서빙 자원을 약한 라벨로 활용 | 서빙 불가 신호(후행 지표)를 라벨에만 사용 |
| 34 | Deep Entity Classification: Abusive Account Detection for Online Social Networks | Xu et al. (Facebook) | 2021 | USENIX Security | [PDF](https://usenix.org/system/files/sec21-xu-teng.pdf) | 이웃 집계 피처 + 저정밀 대량/고정밀 소량 다단계 학습 | 라벨 품질 격차를 학습 구조로 흡수 |

### 3.6 업계 문서

| # | 제목 | 소속 | 연도 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|
| 35 | How does Google prevent invalid activity? / About invalid traffic | Google | 상시 | [링크](https://www.google.com/ads/adtrafficquality/invalid-click-protection.html), [헬프](https://support.google.com/google-ads/answer/11182074) | 필터 + ML + 수동 리뷰, 사후 크레딧 | 다층 구조와 지연 탐지 정산 |
| 36 | Building a healthy open internet by managing traffic quality / About invalid traffic | Criteo | 미확인 | [블로그](https://criteo.com/blog/building-a-healthy-open-internet-by-managing-traffic-quality), [헬프](https://help.criteo.com/kb/guide/en/about-invalid-traffic-zTVZnNIuix/Steps/3151638) | 실시간 룰 엔진 + 오프라인 ML 엔진 + 외부 리스트 | DSP의 실시간/오프라인 분리 |
| 37 | Protect360 anti-fraud guide | AppsFlyer | 상시 | [가이드](https://support.appsflyer.com/hc/en-us/articles/218254203), [블로그](https://www.appsflyer.com/blog/under-the-radar-fraud/) | 유형별 탐지 신호, 실시간 + post-attribution | cTag 단위 피처 목록의 근거 |
| 38 | Fraud Prevention Suite | Adjust | 상시 | [링크](https://www.adjust.com/product/fraud-prevention/), [용어집](https://www.adjust.com/glossary/mobile-ad-fraud/) | 4종 필터 (클릭 인젝션, 분포 모델링, 익명 IP, SDK 서명) | CTIT 분포, 익명 IP |
| 39 | The Methbot Operation | White Ops (HUMAN) | 2016 | [PDF 사본](https://lunio.ai:443/resources/WO_Methbot_Operation_WP.pdf) | 데이터센터 기반 영상 광고 봇넷 | 데이터센터 IP + 스푸핑 조합 |
| 40 | Disrupting PARETO CTV Fraud | HUMAN | 2021 | [PDF](https://www.humansecurity.com/hubfs/HUMAN_Report_Disrupting-PARETO-CTV-Fraud.pdf) | 모바일 봇넷이 CTV 앱을 스푸핑 | 선언 매체와 실제 기기 불일치 |
| 41 | Fraud Lab 사례 (ShadowBot 등) | DoubleVerify | 2022~2025 | [링크](https://doubleverify.com/?p=10881265) | 에뮬레이터·앱 ID 스푸핑 사례 | 방법론은 비공개 |

### 3.7 최근 동향 (2023~2026)

#### 결론

- 최근 3년 안에 "광고 플랫폼이 로그로 어뷰징을 탐지하는 모델"을 다룬 주요 학회 논문은 드묾
  - 확인된 것은 Amazon의 SLIDR(#15, 2023)와 IP 탐지(#16, 2023)뿐. 둘 다 이미 명단에 있음
  - 2024~2026의 Alibaba, Tencent, ByteDance, Meta 발 탐지 모델 논문은 찾지 못함
- 최근 연구의 무게는 세 방향으로 이동
  - 공급망·기여 조작의 측정 (ads.txt 악용, attribution laundering)
  - LLM 활용 (Google의 IVT 탐지)과 LLM 에이전트라는 새 봇 유형
  - 기존 검증 업체의 pre-bid 필터가 뚫린다는 외부 감사
- 아래 자료는 등급을 매기지 않음
  - 이유: 인용이 쌓일 시간이 없어 3.0의 기준으로는 모두 제외됨. 인용 수도 조회하지 못함 (Semantic Scholar 속도 제한)
  - 용도: 정독 대상이 아니라 "지금 무엇이 문제인가"의 확인용

#### 목록

| # | 제목 | 저자/소속 | 연도 | 발표처 | 링크 | 요약 | 시사점 |
|---|---|---|---|---|---|---|---|
| 42 | Unveiling Collusion-Based Ad Attribution Laundering Fraud: Detection, Analysis, and Security Implications | Zhu et al. (소속 미확인) | 2024 | CCS | [PDF](https://donnod.github.io/assets/papers/ccs24_zhu.pdf), [DOI](https://dl.acm.org/doi/10.1145/3658644.3670314) | 여러 앱이 공모해 광고가 다른 앱에서 노출된 것처럼 기여를 세탁하는 수법. RTB의 기여 메커니즘을 정적·동적 분석 | 선언 매체(slot)와 실제 노출 앱의 불일치. 매체 평판 기반 필터를 우회하는 유형. 초록 미열람 |
| 43 | Welcome to the Dark Side: Analyzing the Revenue Flows of Fraud in the Online Ad Ecosystem | Papadogiannakis, Kourtellis, Papadopoulos, Markatos (FORTH, Telefonica Research) | 2025 | WWW (2차 자료 기준) | [arXiv](https://arxiv.org/pdf/2306.08418) | 수천 개 웹사이트를 수집해 ads.txt·sellers.json이 수익 흐름을 숨기는 데 악용되는 양상을 분석 | 공급망 표준만으로는 매체 사기를 못 막음. 판매자 ID 공유 관계가 그래프 간선 후보 |
| 44 | Using AI to fight invalid ad traffic | Google (Ad Traffic Quality, Google Research, DeepMind) | 2025 | 공식 블로그 | [블로그](https://blog.google/products/ads-commerce/using-ai-to-fight-invalid-ad-traffic/) | LLM으로 앱·웹 콘텐츠와 광고 배치를 분석해 기만적 광고 게재를 탐지. 사람 검수를 거쳐 조치 | 로그 통계가 아니라 지면 콘텐츠를 보는 접근. slot 단위 정책 위반 탐지에 해당. 40% 감소 수치는 Google 자체 보고이며 원문 미열람 |
| 45 | On pre-bid bot detection and filtration | Adalytics | 2025 | 업계 보고서 | [보고서](https://adalytics.io/blog/prebid-bot-filtration) | 검증 업체의 pre-bid 필터를 쓰는데도 선언된 봇·데이터센터 트래픽에 광고가 게재된 사례를 보고. 업체들은 반박 | 외부 리스트·벤더에 GIVT 차단을 맡겨도 누수가 있음. 자체 룰 층 유지의 근거. 논쟁 중인 자료 |
| 46 | On the Internet, Nobody Knows You're an LLM Bot: Unmasking Web Agents with Multi-Layer Fingerprinting | Fayolle et al. (소속 미확인) | 2026 | arXiv | [arXiv](https://arxiv.org/abs/2606.30119) | LLM 웹 에이전트 6종을 허니사이트로 유인. 일부는 기존 안티봇을 모두 통과했지만 네트워크·HTTP·브라우저 지문을 결합하면 구분 가능 | 사람처럼 행동하는 새 봇 유형. 행동 패턴 피처만으로는 부족해질 수 있음. 동료 심사 전 |

#### 3년 범위 밖이지만 빠져 있던 것

- BotSpot (CIKM 2020), BotSpot++ (연도·발표처 미확인)
  - 내용: Mobvista의 봇 설치 사기 탐지. 이종 그래프 GNN + GBDT 결합. 일 단위 백만 건 규모로 배포
  - 의미: 우리 접근 (A)+(B)의 결합 형태와 같음. 처음 조사에서 누락됨
  - 상태: 표에는 넣지 않음. 넣으려면 서지·인용 수 확인이 필요
- Self Supervised Pre-training for Large Scale Tabular Data (Amazon, NeurIPS 2022 워크숍)
  - 내용: SLIDR 팀의 후속. 테이블 데이터 자기지도 사전학습으로 봇 탐지 지표를 지도 학습 대비 9% 개선 (자체 보고)

---

## 4. 다음 단계 제안

### 4.1 먼저 확인할 데이터

- 정상 프록시의 양과 분포
  - 질문: 전환이 있는 user key가 전체의 몇 %인가. 슬라이스(OS, 매체 유형, 시간대)별로 비어 있는 곳이 있는가
  - 이유: 정상 라벨 밀도가 낮은 슬라이스는 모델이 과하게 무효화 (SLIDR의 가드레일 이유)
- 룰 적중 로그의 보존 여부
  - 질문: 어떤 룰이, 언제, 어떤 키를 잡았는지 룰별로 남아 있는가
  - 이유: 룰별 라벨링 함수화, 룰 홀드아웃 평가에 필요
- 차단된 키의 사후 데이터
  - 질문: NO_BIDDING 키는 이후 로그가 없는가. bid rate로 일부 통과한 표본이 있는가
  - 이유: 피드백 루프 편향 크기 파악
- 기여 시간 데이터
  - 질문: click→conversion 시간이 cTag(channel) 단위로 집계 가능한가
  - 이유: 클릭 인젝션·스패밍 피처의 전제
- IP 부가 정보
  - 질문: ASN·호스팅 여부·통신사 NAT 구분을 붙일 수 있는가
  - 이유: 그래프 노이즈 제거, GIVT 분리

### 4.2 첫 실험 설계

- 실험 1: slot 단위 GBDT 스코어 (오프라인)
  - 단위 선택 이유: 키 수가 적어 수동 검수가 가능. FDMA 2012와 같은 문제 설정
  - 입력: `daq_ft_ana_*`의 일별 bid/click/conv 집계 + 유저 구성 피처(신규 키 비율, 상위 집중도, 고유 IP 수)
  - 라벨: 양성 = 룰 적중 slot, 음성 = 전환 품질 상위 slot, 나머지 제외
  - 분할: 시간 분할 (앞 기간 학습, 뒤 기간 검증)
  - 평가
    - 룰 홀드아웃 복원율
    - 스코어 상위 구간 중 룰 미적중 slot을 표본 검수 (새로 찾은 것의 정밀도)
    - 스코어 구간별 후행 지표(CVR, 전환 후 이벤트) 단조성
  - 성공 기준: 룰이 못 잡은 slot 중 검수에서 어뷰징으로 확인되는 것이 유의미하게 나옴
- 실험 2: co-visitation 군집 (오프라인)
  - 입력: 기존 `daq_graph_cobid_media_ip_*`
  - 방법: 공용 IP 제거 → 매체 간 중복 비율 간선 → 임계 후 연결요소/밀집 부분그래프
  - 평가: 군집 소속 매체의 후행 지표를 비소속 매체와 비교. 실험 1 스코어와의 상관
  - 산출물: 군집 소속 여부·군집 크기를 실험 1의 피처로 추가했을 때 개선폭
- 실험 3: user key / IP 단위 확장
  - 시점: 실험 1의 평가 체계가 작동한 뒤
  - 변경점: 음성 = 전환 있는 키 (PU 설정), 평가는 IVR·FPR 프록시·robotic coverage
- 온라인 검증
  - 1단계: shadow 스코어 로깅만 (액션 없음)
  - 2단계: 버킷 일부에 bid rate 축소 적용. 광고주 측 CVR·전환당 비용, 매출 손실 비교
  - 3단계: 구간별 eCPM cap → 차단으로 확대

### 4.3 결정이 필요한 사항

- 모델의 1차 목표
  - 룰이 못 잡는 어뷰징 발굴 (보완) vs. 룰 유지보수 비용 절감 (대체)
  - 권장: 보완부터. 대체는 평가 체계가 선 뒤
- 정상 프록시의 정의
  - 전환만 볼지, 후행 이벤트까지 볼지. 전환 사기가 섞이면 정상 라벨이 오염됨
- 수동 검수 자원
  - 주당 검수 가능한 건수. 이 숫자가 평가의 신뢰도를 결정

### 4.4 이번 조사의 한계

- 원문을 열람하지 못하고 초록·2차 요약으로 확인한 항목이 있음 (표에 "본문 미열람", "미확인" 표시)
- Tencent, Twitter/X, IAS는 광고 어뷰징 탐지 방법을 다룬 공개 자료를 찾지 못함
- 2024~2026의 탐지 모델 논문은 거의 찾지 못함. 검색 한계일 수 있으므로 KDD·CIKM·WWW 프로시딩을 직접 검색해 볼 필요가 있음
- 2.5절의 스코어–액션 매핑, 4절의 실험 설계는 레퍼런스에서 유도한 제안이며 직접 인용이 아님
