# 예측기 랜딩 과제 정리
## 2. 콜드스타트 임베딩 초기화 고도화

랜딩 과제로 1에 선행해서 진행한다. (예측기 구조 및 환경 파악)
- 이전 실험 문서: https://wiki.daumkakao.com/x/Fs3Ccw
- wiki 문서: https://wiki.daumkakao.com/x/56UIhg

### 2.1 As-Is in TorchDNN
torch-dnn 콜드스타트 코드 흐름
```
run_online_cvr.py  build_model()                          ┐ config.hyper_parameters.training_method
  └ models/addfommodel.py  AdDFOMModel(config)            ┘ (기본 "AdDFOMModel")
      └ models/model/adsimple.py  AdSimple(config).build() ┐ config.model.mainFF.type
          └ models/model/feature.py  OneHotFeature(feat)   ┘ feat_info["feature_type"] = "one_hot"
              └ nn.Embedding(...)
```

현재 DFM/DFOM 온라인 학습모델의 신규 광고그룹 임베딩 처리
인덱서가 아직 번호를 주지 않은 광고그룹은 group_id가 unknown index인 1로 들어오고, 임베딩 테이블의 1번 행 하나를 같이 씁니다. 1번 행은 매 run 새로 만들어지지만 inject_old_to_new가 이전 run 값으로 덮어 학습이 이어지므로, 결과적으로 지금까지 거쳐 간 신규 광고그룹들의 평균에 가까운 가중치를 갖게 됩니다. 이후 인덱서가 번호를 발급하면 그 그룹은 자기 행으로 옮겨가는데, 그 행은 한 번도 학습되지 않은 초기값(N(0, 1e-5))이라 이 시점에 예측이 크게 흔들리는 것으로 보입니다.

#### unknown index — 신규 그룹이 매칭되는 행은 시간에 따라 둘이다

`nn.Embedding`은 광고그룹 ID 하나에 행 하나가 대응하는 임베딩 테이블이다. 신규 그룹이 매칭되는 행은 하나가 아니라 **시간에 따라 둘**이다.

**1단계 — 1번 행.** 신규 그룹은 처음에 자기 행이 없다. 인덱서에 등록되기 전이라 **모든 신규 그룹이 1번 행(unknown index) 하나를 같이 쓴다.** 이 행은 비어 있지 않다. 신규 그룹 전체의 트래픽으로 학습되어 "신규 그룹 평균" 벡터가 되어 있다. 이전 CTR 실험(ADRECALG-3660·3860)이 건드린 것이 이 행이다. 1번 행을 학습된 값 대신 전체 행의 (가중) 평균으로 갈아 끼웠고, 결과가 지면마다 갈렸다.

**2단계 — 자기 행.** 인덱서가 그 그룹에 번호를 주면 그때 자기 행이 생기는데, **이 행이 "아무 학습된 수치가 없는 행"**이다. $N(0, 10^{-5})$ 로 시작한다. 그 순간 그룹은 학습된 1번 행에서 빈 자기 행으로 옮겨가고, 이 점프가 회의 메모의 과대예측이다. **이전 실험은 이 행을 건드리지 않았다.**

#### 두 단계가 동작하는 코드 (torch-dnn)

##### 1단계 — 신규 그룹이 1번 행을 공유하며 학습하는 경로

**(a) 사전에 없는 ID는 1번** — `utils/vocab.py` 483~485행

```python
output_vocab[k][None] = 0   # 값 없음 → 0번
output_vocab[k][0] = 0
output_vocab[k][-1] = 1     # 모르는 값(-1) → 1번  ← unknown
```

배치 라인이 인덱스 파일을 읽을 때 이 세 줄로 예약 자리를 박는다. 온라인 라인은 Sequoia가 같은 규약으로 인덱스를 매겨 보낸다. 신규 그룹은 여기서 전부 1번이 된다.

**(b) 1번 행은 보통 행처럼 학습** — `models/model/feature.py` 208~214행

```python
self.embed = nn.Embedding(self.input_dim + 1, self.output_dim, padding_idx=feature_config.get("padding_idx"))
torch.nn.init.normal_(self.embed.weight, RN_MEAN, RN_STDDEV)   # 전 행 N(0, 1e-5)
self.embed._fill_padding_idx_with_zero()                       # 0번 행만 0 고정

def forward(self, input):
    return self.embed(input)                                   # 1번이 들어오면 1번 행이 나가고, 역전파도 1번 행에 쌓임
```

`padding_idx`는 0이라 0번만 학습에서 빠진다. 1번을 막는 코드는 없다. 신규 그룹 클릭이 전부 1번으로 들어오므로 1번 행은 신규 그룹 전체의 평균을 배우게 된다.

**(c) run이 바뀌어도 1번 행은 그대로 이어짐** — `utils_online/export.py` `inject_old_to_new`

```python
if "embed" in key and len(shape_before) == 2 and shape_before[0] != shape_after[0]:
    _resize_emb_layer_size(...)          # 테이블 크기가 바뀐 경우 → 2단계
else:  # just copy as before
    value.index_copy_(0, torch.arange(len(old_trainer["state_dict"][key])), old_trainer["state_dict"][key])
```

30분마다 새 run이 뜰 때 이전 모델의 행을 그대로 복사한다. 1번 행도 여기 포함되어 계속 누적 학습된다.

##### 2단계 — 자기 번호를 받은 그룹이 빈 행에서 시작하는 경로

**(a) 테이블이 커지면 늘어난 행은 빈 값** — `utils_online/export.py` `_resize_emb_layer_size`

```python
value.data = torch.nn.Parameter(torch.randn(shape_after))
torch.nn.init.normal_(value.data, RN_MEAN, RN_STDDEV)         # ① 새 테이블 전체를 N(0, 1e-5)로

key_length = min(len(old_trainer["state_dict"][key]), len(value))
index = torch.arange(key_length)
new_value = value * (1 - forget)
value.index_copy_(0, index, new_value[:key_length])
value.index_add_(0, index, old_trainer["state_dict"][key][:key_length], alpha=forget)   # ② 기존 행만 덮어씀
```

인덱서가 신규 그룹에 번호를 주어 상한이 올라가면 이 함수가 돈다. 새 테이블을 전부 $N(0, 10^{-5})$ 로 만들고(①), 기존 번호 범위만 이전 값으로 덮는다(②). **늘어난 꼬리 부분, 즉 방금 번호를 받은 그룹들의 행은 ①의 값 그대로 남는다.** `forget`은 기존 행을 얼마나 물려받을지의 비율이라 새 행과는 무관하다.

이 순간 그 그룹은 학습된 1번 행에서 이 빈 행으로 옮겨진다. 회의 메모의 과대예측 지점이다.

**(b) 은퇴한 번호를 재사용할 때도 빈 값** — `utils_online/export.py` `cleaning_inactive_embed`

```python
not_used = feat_info.get("inactive_values", [])              # 인덱서가 알려준 비활성 번호
...
value[idx] = torch.randn_like(value[idx]) * RN_STDDEV        # 그 행을 N(0, 1e-5)로 리셋
optimizer.state[value]["exp_avg"].index_fill_(feat_dim, idx, 0.0)      # Adam 1차 모멘트 0
optimizer.state[value]["exp_avg_sq"].index_fill_(feat_dim, idx, 0.0)   # Adam 2차 모멘트 0
```

테이블을 늘리지 않고 안 쓰는 번호를 새 그룹에 재배정하는 경우다. 값만 아니라 옵티마이저 상태까지 0으로 밀어서, 새 그룹은 완전한 백지에서 시작한다.

##### 제안이 손댈 자리

| 단계 | 코드 | 지금 | 바꿀 것 |
|---|---|---|---|
| 1 | `feature.py` 1번 행 | 신규 그룹 전체 평균 | 이전 CTR 실험이 여기를 가중 평균으로 교체 |
| 2-a | `_resize_emb_layer_size` ① | 새 행 $N(0, 10^{-5})$ | **형제 그룹 평균**으로 채움 |
| 2-b | `cleaning_inactive_embed` | 재사용 행 리셋 | 같은 값으로 채움. 빠뜨리면 재사용 슬롯을 받은 그룹만 여전히 백지 |

2-a와 2-b는 같은 함수를 호출하게 묶으면 된다. 두 곳 모두 "이 번호가 어느 계정·캠페인의 그룹인가"를 알아야 하므로, 형제 평균을 계산하려면 그룹→계정 매핑을 run 시점에 조회할 수 있어야 한다. 이 부분이 구현의 핵심 의존성이다.

#### 기존/신규 그룹의 구분 기준

기준은 `group_id` feature의 값이다.

| 그룹 | group_id 값 | 임베딩 행 |
|---|---|---|
| 신규, 번호 없음 | 1 | 1번 행 (신규 평균) |
| 신규, 번호 받음 | 예: 10,001 | 자기 행 (빈 값) |
| 기존 | 예: 4,217 | 자기 행 (학습됨) |

두 번째와 세 번째는 값 형태가 같아 구별이 안 된다. 가르는 기준은 "그 행이 한 번이라도 학습됐는가"이고, `exp_avg_sq`가 0인지로 판별한다. 같은 문제가 `account_id`, `track_id`에도 있다. 인덱서와 0·1 규약이 같아서 같은 코드로 덮을 수 있다.


#### 신규 그룹 임베딩이 겪는 두 상태

**Q1. `build_model`마다 테이블이 N(0, 1e-5)로 초기화되는 게 맞나?**

맞다. 다만 바로 다음 `load_prev_model`이 이전 run 값으로 덮는다. 초기화 값이 살아남는 행은 셋뿐이다.

| 행 | 이유 |
|---|---|
| 이전 테이블보다 뒤 | 복사할 값이 없음 |
| 이전에도 있었지만 아무도 안 쓴 행 | 이전 값 자체가 초기화 값 |
| `inactive_values` 행 | `cleaning_inactive_embed`가 다시 밀어버림 |

**Q2. 신규 그룹은 어느 경우인가?**

시간에 따라 두 상태를 거친다.

- 상태 1: 번호 없음 → 1번 행(unknown). 세 경우 어디에도 안 속하고, 이전 값을 물려받는 보통 행.
- 상태 2: 번호 받음 → 자기 행. 인덱서가 준 번호에 따라 위 세 경우 중 하나. 어느 쪽이든 빈 값.

**Q3. 1번 행이 "신규 그룹 평균"이 되는 이해가 맞나?**

맞다. 인덱서가 번호 없는 그룹을 1로 쓰고, 1번은 `padding_idx`가 아니라 학습·복사되는 보통 행이며, 매 run 신규 그룹 클릭이 전부 여기 쌓인다. 정확히는 Adam 때문에 최근 신규 그룹 쪽으로 기운 시간 가중 평균이다.

**Q4. 상태 1 → 2는 누가 바꾸나?**

Sequoia 인덱서. `sequence`를 올려 번호를 발급하면 이후 로그부터 1 대신 그 번호가 적힌다. torch-dnn은 두 값이 같은 그룹인지 모른다.

**Q5. 상태 1의 가중치는 전달되나?**

안 된다. 1번 행에 그 그룹 몫이 따로 없고, torch-dnn은 새 번호와 1번의 연결을 몰라 번호 같은 행끼리만 복사한다. 예측은 "신규 그룹 평균" → "빈 임베딩"으로 점프하고, 이 지점이 회의 메모의 과대예측이다.

**Q6. 온라인 학습에서 신규 광고그룹 정의**
온라인 학습 (run_online_cvr.py와 그 하위)에는 신규 광고그룹을 판정하는 부분을 찾을 수 없음. 어디서 정의해야 하나?

#### 의문

**의문 1. 왜 상태 1을 거쳐 상태 2로 오나**

torch-dnn엔 답이 없고 Sequoia 인덱서 설정에 있다. 후보는 둘이다.

- 발급 지연: 인덱서 갱신이 주기적이라 첫 클릭부터 다음 갱신까지 로그가 1로 남음.
- 의도적 임계치: 클릭 N건 이상일 때만 번호를 줘서 일회성 ID로 테이블이 부푸는 걸 막음.

어느 쪽이든 클릭 즉시 발급은 어렵다. 어느 쪽인지는 Sequoia 담당자에게 확인이 필요하다.

> **답변 (담당자).** 발급 지연이 맞고, 의도한 설계는 아니다.
> - 인덱서는 광고그룹이 로그에 처음 잡히면 바로 번호를 준다 (threshold 1).
> - 다만 인덱서 갱신이 1분 주기라, 그 사이에 들어온 노출은 1로 들어간다. 갱신 시각은 이그드라실의 `ctime`에서 볼 수 있다.
> - 인덱서가 로그를 읽어 번호를 매기는 구조라 로그에 먼저 찍혀야 번호가 생긴다. 1을 거치게 한 이유가 따로 있는 게 아니다.
>
> **함의.** 상태 1 구간이 첫 노출 후 1분이다. CVR 학습 row는 클릭이므로 그 1분 안의 클릭만 1로 들어오고, 사실상 없다고 봐야 한다. 신규 그룹은 처음부터 빈 자기 행에서 시작하는 셈이고, 문제는 "학습된 1번 행 → 빈 행 점프"가 아니라 **빈 행에서 학습이 덜 된 채로 서빙되는 기간** 하나로 단순해진다. 실제 비중은 온라인 run 로그 `print_special_index`의 `group_id(1)` 비율로 확인할 수 있다.

**의문 2. torch-dnn이 신규 여부와 소속을 아나**

| | 상태 1 | 상태 2 |
|---|---|---|
| 신규인가 | 안다 (값이 1) | 모른다. "처음 등장한 번호"를 직접 찾아야 함 |
| 계정 | 안다. 같은 row에 `account_id` feature가 있음 | 같음 |
| 캠페인 | 모른다. `campaign_type`만 있고 campaign_id 없음 | 같음 |

번호를 받은 뒤에는 모델 입장에서 신규 그룹과 기존 그룹을 구별할 방법이 있는가?(unknown_index 면 신규다?)


> **답변 (담당자).** 번호만으로는 신규와 기존을 가를 수 없어서, 분석할 때 기준을 따로 정해야 한다. 참고할 이전 기준 둘:
> - 소재 임베딩 실험: 소재 단위로 테스트일 노출 1천 미만을 콜드로 절단. https://kakao.atlassian.net/wiki/pages/viewpage.action?pageId=144509681
> - 광고그룹 pCVR cal 수렴 분석: 직전 3주 노출 0 → 첫 등장을 신규로 정의. https://kakao.atlassian.net/wiki/pages/viewpage.action?pageId=144509087
>
> **함의.** 노출수는 DFM/DFOM 파이프라인(`cvr-*-dfom-kuid-v1`)에는 없다. row가 클릭이라 세어도 클릭수다. 노출수를 세려면 CTR 파이프라인(`ctr-talk-bimp-join-v2` 계열, row = 과금 노출 1건)의 row를 group_id별로 세야 하며, group_id 인덱서가 글로벌이라 번호가 CVR 임베딩 행과 그대로 대응한다.

#### 같은 계정·캠페인 형제 그룹 임베딩의 평균 을 하려면?
(config의 feature 필드 참고)  
계정 단위면 학습 배치 row 안에 account_id와 group_id가 나란히 들어오므로 데이터에서 매핑을 직접 만들 수 있습니다. 캠페인 단위면 campaign_id가 feature에 없어서 외부 조회나 feature 추가가 필요합니다.

계정 단위로 할 때 흐름은 이렇습니다.
```
배치 row:  account_id=A, group_id=10001, ...     ← 신규 그룹 G
           account_id=A, group_id=4217,  ...     ← 형제
           account_id=A, group_id=4218,  ...     ← 형제

배치를 읽으며 {group_id → account_id} dict를 쌓습니다. 
_count_index가 이미 row마다 feature 값을 훑고 있어 그 자리에 한 줄 추가하면 됩니다.
G가 처음 등장한 행(10,001)을 찾으면 dict에서 A를 얻고, A에 속한 다른 그룹 번호들의 임베딩 행을 평균해 10,001행에 씁니다.
```

주의할 점이 하나 있습니다. 형제 그룹이 이번 run 배치에도 나타나야 매핑에 잡힙니다. 30분 배치에 A의 그룹이 G 하나뿐이면 형제를 못 찾습니다. 두 가지 선택지가 있습니다.

### 2.2 이전 CTR 실험(ADRECALG-3660·3860)에서 배울 것

| 시점 | 방법 | 결과 |
|---|---|---|
| 2025.11 오프라인 | unknown 행을 전체 행 **단순 평균**으로 교체 | 과예측 **악화** (+29%). 노출 적은 미학습 행들이 평균을 오염 |
| 2025.12 오프라인 | 최근 10분 bimp 비중 **가중 평균** | slot unknown lift 74.7% → 12%, group_id 17% → −0.2% (network) |
| 2026.01 온라인 3일 | 가중 평균, 4피처 | 전체 −12.4% 개선. account_id −46% |
| 2026.02 온라인 7일 | 같음 | 전체 **+4.4% 악화**. tb −30%·criteo −6% 개선 / daum·network +6% 악화 / slot +181% 악화 / group_id +4.9% 악화 |

교훈 셋.
1. **전역 평균은 지면·피처마다 부호가 갈린다.** "지금 트래픽의 평균 광고"가 "신규 그룹의 기대값"과 다르기 때문이다.
2. **3일 → 7일로 가며 효과가 사라졌다.** 고정 벡터로 교체하면 그 뒤 학습이 진행돼도 unknown 행은 낡는다.
3. **account_id 조건이 붙은 구간이 가장 크게 개선됐다**(중간 −46%). 계정 정보가 신호를 갖는다는 증거이고, 아래 형제 그룹 평균의 근거다.

단순 평균 코드는 torch-dnn 태그 `ADRECALG-3660-1`에 `replace_unknown_with_mean`(`utils_online/export.py`)으로 남아 있고 main에는 없다. 설정 키는 `mean_embedding.init_with_mean_embedding`, `replace_unknown_with_mean_embedding`, `unknown_index_features`. **가중 평균 버전은 torch-dnn 어느 ref에도 없다.**

### 2.3 리서치: 초기값을 어디서 가져오나

**전제.** 신규 광고그룹의 임베딩 행을 만들어야 한다는 전제하에서 찾았다. 즉 빈 임베딩 행을 아예 안 만들도록 우회하는 방식은 제외했다. 가령 ID를 해시 버킷에 매핑하는 Hashing trick, 테이블 대신 MLP로 임베딩을 만드는 DHE, ID 임베딩을 콘텐츠 인코더로 대체하는 MoRec, 아이템을 콘텐츠 코드로 표현하는 TIGER, LLM으로 학습 데이터를 만들어 주는 LLM Data Augmenter·ColdLLM이 그것이다. 학습 시작 시 전 행을 위한 초기화(Xavier·He, N(0, σ²))도 별개 문제라 제외했다. 6편이며 인용 수는 OpenAlex 기준이다.

| 연도 | 방법 | 핵심 | 레퍼런스 |
|---|---|---|---|
| 2012 | 계층 추정 | 신규 광고 CVR을 광고주·캠페인 계층의 과거 성과로 추정. 데이터가 적을수록 부모로 수축. **형제 평균의 통계적 원형** | Lee et al., KDD 2012 (Turn), 210회<br>Estimating Conversion Rate in Display Advertising from Past Performance Data |
| 2015 | LightFM | 아이템 벡터 = 메타 feature 임베딩의 합 (+ 선택적 ID 임베딩). 콜드 아이템은 ID 행이 0이라 메타만으로 표현. **계정 + 그룹 잔차의 원형** | Kula 2015, 136회<br>Metadata Embeddings for User and Item Cold-start Recommendations |
| 2019 | MetaEmb | MetaEmb는 외부 생성기가 warm 데이터로 미리 학습해 두었다가, 신규 아이템의 초기 임베딩을 생성해 넣어 주고 이후에는 역할이 없다. 초기값에서 몇 스텝 학습한 뒤 손실이 작도록 메타러닝. "빨리 배우는 출발점". 광고 CTR 대상.  | Pan et al., SIGIR 2019 (Alibaba), 179회<br>Warm Up Cold-start Advertisements: Improving CTR Predictions via Learning to Learn ID Embeddings |
| 2021 | MWUF | MWUF는 콜드 행을 학습된 보정(γ·β)으로 warm 분포 쪽으로 옮겨 예측에 사용한다. 클릭 하나로 손실 2개를 계산해, 콜드 행은 보정 전 예측의 손실로 보통처럼 학습하고, 보정 네트워크는 보정 후 예측의 손실로 학습한다. | Zhu et al., SIGIR 2021, 129회<br>Learning to Warm Up Cold Item Embeddings for Cold-start Recommendation with Meta Scaling and Shifting Networks |
| 2021 | GME | MetaEmb 확장판. MetaEMB가 신규 광고의 속성만 보고 초기 임베딩을 만든다면, GME는 속성이 겹치는 기존 광고들의 정보까지 끌어와 만듭니다. 신규 광고와 속성(카테고리·브랜드 등)이 겹치는 기존 광고를 이웃으로 뽑고, 신규 광고 속성으로 만든 예비 임베딩을 이웃 정보로 GAT가 다듬어 초기 ID 임베딩을 생성. **무작위·이웃 평균 초기화를 베이스라인으로 두어 우리 현재·제안과 직접 비교됨**  | Ouyang et al., SIGIR 2021 (Alibaba), 50회<br>Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction |
| 2022 | VELF | 광고 ID 임베딩을 점이 아니라 분포(평균·분산)로 학습한다. 광고마다 클릭으로 학습되는 ID 분포와, 광고 속성으로 계산하는 기준 분포를 두고, 학습 손실에 두 분포의 거리를 더해 ID 분포를 기준 분포 쪽으로 당긴다. 그 결과 클릭이 많은 광고는 자기 데이터에 맞는 분포를 갖고, 적은 광고는 기준 분포 근처에 머물러 과적합을 피한다. 서빙 때는 두 분포의 평균을 빈도에 따라 섞어, 신규 광고는 기준값에서 시작해 데이터가 쌓일수록 자기 값으로 넘어간다. | Xu et al., WWW 2022, 31회<br>Alleviating Cold-start Problem in CTR Prediction with A Variational Embedding Learning Framework |

**제외한 논문.** 신규 ID 행을 만들고 채운다는 전제에 맞지 않아 표에서 뺐다.

| 논문 | 핵심 개념 | 제외 이유 |
|---|---|---|
| DropoutNet (NeurIPS 2017) | 학습 중 ID 임베딩을 무작위로 0으로 지워, 콘텐츠만으로도 예측하게 훈련 | 초기값이 아니라 훈련법이다. 주 모델을 다시 학습한다 |
| Heater (SIGIR 2020) | DropoutNet의 "0으로 지움"을 "콘텐츠 → MoE 변환 표현으로 대체"로 바꿈 | DropoutNet과 같은 이유 |
| CVAR (SIGIR 2022) | 속성과 ID 임베딩을 VAE의 공용 공간에서 맞춘 뒤 임베딩을 생성 | 속성과 ID 임베딩이 서로 다른 공간이라는 문제를 푼다. 우리가 쓸 형제 임베딩·계정 임베딩은 신규 행과 같은 공간이라 번역이 필요 없고, 광고 속성도 빈약하다 |
| AVAEW (arXiv 2023) | CVAR에 적대적 정렬을 더해, 생성 임베딩이 warm 분포와 닮게 함. 뉴스 플랫폼 온라인 A/B 보고 | CVAR의 공간 번역을 적대적 정렬로 강화한 것이라 같은 이유로 해당 없다. 학회 게재 논문도 아니다 |
| EmerG (KDD 2024, Baidu) | 아이템 속성으로 아이템별 feature 상호작용 그래프를 생성. ID 임베딩은 무작위 초기화 그대로 | 임베딩이 아니라 상호작용 구조를 바꾼다. 모델 구조 변경이다 |
| MeLU (KDD 2019) | MAML로 모델 결정층을 "몇 스텝 적응 뒤 잘 맞기"로 학습 | 학습 목표 자체가 바뀐다 |
| MAMO (KDD 2020) | MeLU의 "모두에게 같은 초기 파라미터" 대신, 메모리로 유저별 맞춤 초기 파라미터를 만듦 | 유저 콜드스타트 대상이고, ID 행이 아니라 모델 파라미터를 초기화한다 |

**논문 유형 정리** 여섯 편 모두 "신규 광고는 자기 데이터 대신 닮은 광고에게서 빌려온다"는 원리를 공유한다. 무엇에게서 빌리는지와 언제 빌리는지로 나누면 다음과 같다.

*1. 무엇에게서 빌리나*

| 빌려오는 곳 | 논문 | 빌리는 방식 | torch-dnn 대응 |
|---|---|---|---|
| 이웃·형제 광고의 임베딩 | GME, 계층 추정 | 이웃 값을 평균하거나 가중해 그대로 씀 | 같은 계정 형제 group_id 행의 평균 |
| 부모(속성)의 임베딩 | LightFM, VELF, MetaEmb, MWUF (γ) | 기준값으로 씀(VELF), 더함(LightFM), 생성기 입력(MetaEmb), 콜드 행 크기 조절(MWUF) | account_id 임베딩, 또는 계정·캠페인 유형 속성으로 만든 기준값 |
| 그 광고를 클릭한 유저 | MWUF (β) | 유저 정보로 콜드 행을 이동(shift) | 그룹을 클릭한 유저 임베딩 평균. 우리 config엔 user ID 임베딩이 없어 약함 |

첫째와 둘째 행은 사실상 같은 정보다. 계정 임베딩에는 형제들의 클릭이 모두 흘러들어 "형제들의 공통 성분"이 담기고, 형제 평균은 그걸 행들에서 직접 계산한 것이다. MWUF만 유저 정보를 함께 쓴다.

*2. 언제 빌리나*

| 시점 | 논문 | 효과 |
|---|---|---|
| 신규 등장 시 한 번, 초기값으로 | MetaEmb, GME, 계층 추정(n=0) | 출발점만 바꾸고 이후는 자기 학습 |
| 학습 중 계속 당김 | VELF (거리 항) | 데이터가 적은 동안 부모 근처에 묶어 둠 |
| 예측할 때마다 섞거나 보정 | VELF (서빙 섞기), MWUF, LightFM (합 구조) | 콜드 구간 내내 작동하고 점프가 없음 |

**베이스라인 등장 빈도.** arXiv 판본이 있는 최신 논문(MWUF, GME, CVAR, DHE, MoRec, TIGER, LLM Data Augmenter, ColdLLM) 8편의 실험 절에서 확인. 표에서 뺀 논문도 비교군 집계에는 포함.

| 베이스라인 | 등장 논문 | 비고 |
|---|---|---|
| MetaEmb (2019) | MWUF, GME, CVAR, ColdLLM | 사실상 표준 비교군 |
| DropoutNet (2017) | MWUF, CVAR, ColdLLM | 표준 비교군 |
| 무작위 초기화 | GME(RndEmb), MWUF(암묵적 하한) | 우리 현재 상태 |
| 전역 평균 초기화 | MWUF (구성요소) | "무작위보다 낫다"고만 서술. 이전 CTR 실험이 이것 |
| 이웃 평균 초기화 | GME (NgbEmb) | **형제 평균과 같은 발상.** 결과는 "MetaEmb보다 나을 때도 못할 때도 있어 단순 평균은 그리 효과적이지 않다" |
| MWUF (2021) | CVAR | |
| GAR, ALDI (2022~23) | ColdLLM | |


**우리 제안에 대한 시사점.**
- 형제 평균 채우기는 GME가 NgbEmb로 이미 베이스라인에 넣었고 결과가 들쭉날쭉했다. 다만 GME의 이웃은 속성 유사도로 고른 광고이고 우리는 같은 계정 그룹이라 이웃의 질이 다르다. 1차 실험 결과를 NgbEmb 결과와 나란히 두면 위치가 분명해진다.
- "다음 단계"의 표준 비교군은 MetaEmb와 DropoutNet이다. 형제 평균 뒤에 생성기로 가려면 MetaEmb를 구현해 비교군으로 두는 게 문헌과 맞추는 길이다.
- 우리 위치: 형제 평균은 Lee 2012의 임베딩판, 후속인 계정 + 잔차는 LightFM 구조, 그 다음 후보는 MWUF. LLM 계열은 소재 텍스트·이미지가 파이프라인에 들어와야 의미가 있어 범위 밖.

### 2.4 고도화 방향

**용어.** 임베딩 테이블은 group_id 번호 하나에 행 하나가 대응한다.
- **1번 행**: 인덱서가 아직 번호를 주지 않은 그룹이 공통으로 쓰는 unknown index(=1)의 행. 여러 신규 그룹의 클릭이 모두 여기로 들어와 학습되므로 "신규 그룹 평균"에 가까운 값을 갖는다. 발급 지연이 1분이라 CVR에서는 거의 쓰이지 않는다.
- **빈 자기 행**: 인덱서가 번호를 발급해 그 그룹 전용으로 생긴 행. 아직 그 그룹의 클릭으로 한 번도 갱신되지 않아 초기값 N(0, 1e-5) 그대로다. 신규 그룹은 사실상 여기서 시작하며, 과대예측이 나는 구간이다.

**핵심 주장.** 이전 실험의 실패 원인은 평균을 낸 것이 아니라 **조건 없이** 평균을 낸 것이다. 광고주 간 CVR은 몇 배씩 다르므로 전역 평균은 모두를 같은 값으로 회귀시킨다. 같은 계정 형제 그룹의 평균은 그 광고주의 CVR 수준을 담는다. 이것은 신규 광고 CVR을 광고주·캠페인 계층의 과거 성과로 추정한 Lee et al. (KDD 2012)의 임베딩판이다.

- **처방.** 빈 자기 행을 **같은 계정 형제 그룹 임베딩의 평균**으로 채운다. 형제가 없으면 1번 행 값으로 폴백. 1차는 단순 평균, 2차에서 형제의 갱신 횟수·클릭 수로 가중. 캠페인 단위는 campaign_id feature가 없어 이번엔 제외.
- **캘리브레이터와의 관계.** 서빙 캘리브레이터가 그룹별 실측/예측 비율을 곱하는 방식이면 신규 그룹은 실측이 없어 캘리브레이터도 도울 수 없는 구간이고, 초기화는 그 앞단이다. 겹친다면 캘리브레이터가 계정 단위 prior를 쓰는 경우인데, 위치 확인 후 판단.

| | 이전 CTR 실험 | 이번 제안 |
|---|---|---|
| 건드리는 행 | 1번 행 (번호 없는 신규 그룹들이 공유하는 unknown 행) | 빈 자기 행 (번호는 받았지만 아직 학습되지 않은 그룹 전용 행) |
| 채우는 값 | 전체 행의 가중 평균 | **같은 계정 형제 그룹 행의 평균** |
| 왜 다른가 | 모든 신규 그룹에 같은 벡터 → 광고주 간 CVR 차이를 못 담음 | 그 광고주의 다른 그룹들 값 → CVR 수준이 비슷할 가능성 높음 |

이전 실험은 "효과가 없었다"가 아니라 **지면과 피처마다 부호가 갈렸다.** tb는 −30%로 크게 좋아졌고 daum·network는 +6% 나빠졌다. 전체 평균 하나로 모두를 덮으니 맞는 곳과 틀리는 곳이 갈린 것이고, 그게 조건을 붙여야 한다는 근거다.

한 줄로 하면, **"빈 자기 행을 전체 평균으로 채우면 반은 맞고 반은 틀리니, 그 그룹의 형제들 평균으로 채우자"**가 제안이다.

#### ① 형제 그룹 사용 — 신규 행 초기화 (A, B)

신규 행에 처음부터 형제 정보를 써 넣는다. 새 모듈도 서빙 그래프 변경도 없다.

**등장 횟수.** B에서 쓰는 "학습 정도"는 학습 로그에 그 group_id가 들어온 줄 수다. 지금 run의 배치 안에서만 세거나, 정해진 시점부터 누적한다. 누적은 모델 안에서 배치마다 세어 run 간 이어받거나, 과거 로그에서 집계해 run 시작 시 읽는다.

**안 A — 형제 평균을 한 번 채움**

- **핵심:** 처음 등장한 신규 행을 같은 계정 형제 행들의 단순 평균으로 한 번 채우고, 이후는 평소처럼 학습한다. 형제가 없으면 1번 행 값을 쓴다. 가장 싸지만 덜 학습된 형제도 같은 비중으로 들어가고, 도움이 첫 순간뿐이다.
- **구체적인 구현:** `AdDFOMModel`에 학습 배치 시작 훅을 추가해, 학습된 적 없는 group_id 행을 형제 평균으로 채운다.
- **레퍼런스:** Lee et al., KDD 2012 (Turn), *Estimating Conversion Rate in Display Advertising from Past Performance Data*. Ouyang et al., SIGIR 2021 (Alibaba), *Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction*의 NgbEmb 비교군.

**안 B — 잘 학습된 형제를 더 믿고 평균**

- **핵심:** A의 평균을 등장 횟수에 따른 가중 평균으로 바꿔, 막 생긴 형제가 평균을 흐리지 않게 한다. 이전 CTR 실험의 단순 평균이 미학습 행에 오염돼 실패한 것에 대한 대응이다.
- **구체적인 구현:** A의 훅에서 평균을 가중 평균으로 바꾼다. 등장 횟수는 배치 안에서 세거나 누적하며, 1차는 옵티마이저 상태로 대신해도 된다.
- **레퍼런스:** GME(위)의 이웃 가중. Huang et al., SIGIR 2023, *Aligning Distillation For Cold-start Item Recommendation*의 교사 가중.

#### ② 부모(account_id) 임베딩 사용 (D)

형제 평균을 계산하는 대신, 형제들의 공통 성분을 이미 담고 있는 계정 임베딩을 그대로 쓴다.

**안 D — 그룹 임베딩 = 계정 임베딩 + 그룹 잔차**

- **핵심:** group 자리에 그룹 행 대신 계정 행과 그룹 행의 합을 넣는다. 신규 그룹은 그룹 행 ≈ 0이라 자동으로 계정 값에서 시작하고, 판별·섞기·형제 평균 계산이 필요 없으며 계정 행이 계속 학습돼 값이 낡지 않는다. 기존 모델을 옮기지 않고 이 구조로 처음부터 다시 학습한 새 모델로 교체한다.
- **구체적인 구현:** `OneHotFeature`와 `AdSimple`의 forward가 group_id에 account_id 행을 더하도록 바꾸고, config에서 두 임베딩 차원을 같게 맞춘다. 새 config로 처음부터 학습해 기존 모델과 A/B로 비교한다.
- **레퍼런스:** Kula 2015, *Metadata Embeddings for User and Item Cold-start Recommendations* (LightFM).

### 2.5 실험 설계

**목적** 신규 광고그룹이 빈 행 N(0, 1e-5)에서 시작하는 구간의 과대예측을 줄인다. 빈 행을 같은 계정 형제 그룹의 임베딩 평균으로 채운다.

**신규 판별** Adam `exp_avg_sq`가 0인 행. 한 번도 역전파를 받지 않은 행이라 외부 데이터 없이 판별된다. torch-dnn에 없는 새 로직이며, 현재 코드는 `cleaning_inactive_embed`에서 이 값을 0으로 쓰기만 한다.

**골격** `run_online_cvr.py` `train_model`, `load_prev_model` 뒤 fit 전.

```
1. 신규 행 찾기    new_rows = exp_avg_sq[group_id 테이블].sum(1) == 0
2. 매핑 만들기     이번 run 배치 row에서 {group_id → account_id}
3. 형제 찾기       같은 account_id인 다른 group_id들
4. 평균 내기       형제 임베딩 행 평균 (빈 형제 제외)
5. 채우기          embed.weight[new_row] = 그 평균
```

형제가 없으면 1번 행 값으로 fallback. 채운 행의 Adam 상태는 0으로 둔다.

**비교군** A 현재 / B 1번 행 값으로 채움 / C 형제 평균. B가 있어야 "형제 효과"와 "빈 행만 아니면 되는 효과"가 갈린다.

**평가** 오프라인 재생. 신규 기준은 담당자 기준(노출 N 미만 또는 3주 무노출 후 첫 등장)을 Hive 집계로 적용. 주지표 캘리브레이션, 보조 RIG·AUC. 첫 등장 후 1h/6h/1d/3d 경과별로 본다.

**로그** 신규 행 수, 형제 매칭·fallback 비율, `group_id(1)` 비율.

**한계** 형제는 계정 단위(campaign_id 없음). 이번 run에 나타난 형제만 잡힘. group_id만 다룸.

### 2.6 평가 방법

효과 확인은 오프라인에서 신규 그룹만 걸러 지표를 본다. 신규 기준은 온라인 판별(`exp_avg_sq`)과 달라도 되고, 오히려 달라야 한다. 모델 자신의 기준으로 자르면 "내가 고른 행에서 내가 좋아졌다"가 되어 설득력이 없다.

| 항목 | 결정 |
|---|---|
| 신규 기준 | 담당자 기준 그대로. "테스트일 노출 N 미만" 또는 "직전 3주 노출 0 → 첫 등장". Hive에서 그룹별 노출 집계 |
| 지표 | 주지표 캘리브레이션(pCVR 합 / 전환 합). 보조 RIG·AUC |
| 시간 축 | 첫 등장을 0으로 두고 1h / 6h / 1d / 3d 경과별. "얼마나 빨리 1에 붙나" |
| 비교군 | A 현재 / B 1번 행 값 / C 형제 평균. 같은 기간 |
| 방식 | 오프라인 재생 먼저. 세 군이 같은 데이터를 봄. 차이가 확인되면 온라인 A/B |

배치 라인의 `--eval_cold` 틀을 빌려 온라인 결과에 신규 필터를 붙이면 빠르다.

### 2.7 확인 필요

- 2025.12 **가중 평균 구현 코드**의 위치 (torch-dnn에는 단순 평균만 남음)
- 회의의 **캘리브레이터**가 무엇이고 어디 있는지
- **인덱서 갱신 주기** — 새 group_id에 인덱스가 붙는 시점을 정하는 외부 작업. 단계 A의 길이가 여기에 달림

### 2.8 기타

#### 초기 제안 내용
**의미 있는 값으로 초기화하면 초기 pCVR 안정에 도움이 될 수 있다.** 초기값 후보는 아래 순서로 본다.

| | 초기값 | 평가 |
|---|---|---|
| 1 | 전역 group_id 임베딩 평균 | 사실상 무효 — 지금도 예측이 "평균적인 그룹" 값이라 달라지는 게 없다 |
| 2 | **같은 계정·캠페인 형제 그룹 임베딩의 평균** | 싸고 정보가 있다. 형제가 없으면 계정 → 목적 순으로 폴백. **첫 실험감** |
| 3 | 속성(계정·캠페인·목적·소재)을 받아 임베딩을 생성하는 모델 | MetaEmb 방식. 가장 강하지만 학습 파이프라인이 하나 늘어난다 |

**구현 시 주의 두 가지.** ① 초기화 지점이 한 곳이 아니다 — 모델 생성 시점의 `torch.nn.init.normal_` 외에 `utils_online/export.py`의 슬롯 재사용 경로(`reset not used embedding`)와 `cmd/reset_embed.py`도 같이 고쳐야 한다. 안 그러면 은퇴한 슬롯을 물려받은 새 그룹은 여전히 $10^{-5}$ 로 시작한다. ② 옵티마이저 모멘트를 0으로 민 상태에서 출발값만 커지면 초반 갱신 동역학이 바뀐다.

**측정**: 초기화 방식별로 그룹 생성 후 경과 시간별 캘리브레이션 오차 곡선을 겹쳐 그린다. 첫 몇 시간 구간의 차이가 C4의 크기다.

2번 내용 위주로 검증 필요.

#### 회의 내용 (raw)
임베딩 초기화 제안

unknown index 를 안쓰고 형제 임베딩을 그때그때 계산해서 넣어준다. 
임베딩 초기화 기법 괜찮은게 있으면 추가해서 넣어준다. 
캘리브레이터와 겹칠 수도?

뭘 가지고 신규 광고그룹을 정의하러냐? ← 데이터를 가지고 기준을 정해야 한다. 
지금은 노출이 하나도 없으면 신규 광고다. 

지금 문제는 unknown index 모든 신규광고그룹이 쓴다는 것
신규가 아니게 되는 시점 (노출이 되는 시점) 바로 임베딩을 쓰는데 학습이 덜 되었으니 과대예측이 이루어진다. 
