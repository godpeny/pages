# 위너(PP) 모델이 bid_price를 예측하는 방법: 코드 레벨

torch-dnn `origin/master` 기준이에요. 예시는 `config/dr/all/dr_online_talk_all_v9.7.2.yaml`(pb_talk·pb_talk_bizboard 통합 모델)이에요.

## 0. 한눈에 보기

```
YAML config ──► WpWrapModel (학습 래퍼: loss·metric·optimizer)
                   └─ main_net = WpCrossv2
                        ├─ 피처 유닛들 (임베딩/dense) → concat (78차원)
                        ├─ DCN_v2: cross layer 1층 → Dense 32 + LayerNorm + ReLU → Linear(1)
                        └─ exp(출력) = bid_price
학습: log1p(bid_price) vs log1p(ssp_close_price) smooth L1, 10분 단위 warm start
서빙: 같은 forward + dr/bsr 계산식이 들어간 ONNX
```

| 구성 | 파일 |
|---|---|
| 실행 진입점 | `run_online.py` (`training_method: WpWrapModel`이면 이 래퍼 사용) |
| 학습 래퍼 | `models/wpwrapmodel.py` |
| 본체 네트워크 | `models/model/wpcrossv2.py` (`WpCrossv2`, `DCN_v2`) |
| 피처 유닛 | `models/model/feature.py` |
| loss | `models/modules/loss.py` (`log_smooth_l1_loss`) |
| 데이터 로더 | `data_loader/online/sequoia_dataset.py` |
| warm start | `utils_online/export.py` (`inject_old_to_new`) |

## 1. 모델 종류

위키 분류로는 **Price Prediction(PP)**이고, 가격을 바로 회귀하는 방식이에요. 네트워크는 **DCN v2**(Deep & Cross Network v2, Google 2020)를 단순화한 구조예요.

`models/model/`에는 변형이 몇 개 더 있어요. pb_* 경쟁 모델 config는 모두 기본형인 `mainFF.type: WpCrossv2`를 써요.

네 클래스는 **입력부(피처 유닛 → concat)와 DCN v2 본체가 같아요.** 다른 건 **네트워크 출력을 어떻게 가격으로 바꾸는지**, 그리고 그에 맞는 loss예요.

| 클래스 | 네트워크 출력 | 가격(bid_price) 계산 | 짝이 되는 loss | 현재 사용처 (`config/dr` 기준) |
|---|---|---|---|---|
| **WpCrossv2** | 값 1개 | `exp(out)` | `log_smooth_l1_loss` | **pb_* 전부** (daum, network, service, talk_all) |
| **WpCrossv2Binning** | 값 128개 (버킷별 점수) | 버킷 점수에 softmax 등을 씌운 뒤, 각 버킷 대표값의 가중합 | 버킷 분류 loss | 현재 없음. `config/legacy`의 옛 talk(v9.5.x), tb(v9.10.0)에서 사용 |
| **WpCrossv2LogNorm** | 값 2개 (μ, σ) | 기본은 중앙값 `exp(μ)`. 평균·최빈값도 선택 가능 | `censored_loss` | **criteo, adpopcorn** (외부 SSP) |
| **WpCrossv2MTL** | 가격 헤드 1개 + 경쟁 DSP 수 헤드 | `exp(out)` (가격은 WpCrossv2와 같음) | `mtl_*_nll` | 현재 없음 (실험용) |

### WpCrossv2: 기본형

가격 하나를 바로 맞혀요. 이 문서의 2~5절이 이 구조를 설명해요.

### WpCrossv2Binning: 가격을 구간으로 나눠서 예측

- 가격 범위를 128개 구간(버킷)으로 나누고, 정답이 어느 구간에 있을지 구간별 점수를 내요(`Bucketize`).
- 최종 가격은 "구간 확률 × 구간 대표값"의 합이에요.
- 위키에 나오는 "낙찰가 범위를 128개 버킷으로 예측하는 PP"가 이 방식이에요. 지금은 기본형으로 대체됐어요.

### WpCrossv2LogNorm: 가격의 분포를 예측

- 가격 한 점이 아니라 **로그정규 분포의 μ(중심)와 σ(퍼짐)**를 내요.
- criteo 같은 외부 SSP는 진 경매의 낙찰가를 거의 알려 주지 않아요(lurl 1~2%). 대부분 "내 입찰가보다 높았다 / 낮았다"만 알 수 있어요. 이런 **잘린(censored) 정보**를 학습하려면 분포를 예측하는 게 자연스러워요.
  - 낙찰가를 아는 샘플: 그 값이 나올 확률(NLL)로 학습
  - 승패만 아는 샘플: "입찰가보다 높을 확률"(BCE)로 학습
- `bid_optimizer`를 붙이면 분포를 이용해 기대 이익(Surplus)이 최대인 입찰가를 고를 수 있어요. 위키 과제 ⑬(분포 예측 + Surplus 최대 입찰)의 근거인 criteo 실험(Surplus +25%)이 이 구조예요.

### WpCrossv2MTL: 가격 + 경쟁 강도를 같이 학습

- DCN을 **두 개**(`mainFF`, `bidFF`) 둬요.
  - `mainFF`: 가격을 예측해요.
  - `bidFF`: "이 경매에 외부 DSP가 몇 곳 입찰할까"를 예측해요. 이항, 베타이항, 0-과잉 베타이항 중 선택할 수 있어요.
- 경쟁 강도를 보조 과제로 학습시켜 가격 예측을 돕는 멀티태스크 구조예요. 서빙 출력은 기본형과 같아요.

## 2. 입력: 피처 유닛

config의 `features`를 위에서부터 읽어서 피처마다 유닛 하나를 만들어요(`WpCrossv2.build`). `target`과 `auxiliary` 타입은 네트워크 입력에서 빠져요.

| 피처 | feature_type → 클래스 | 처리 | 출력 차원 |
|---|---|---|---|
| gender, age_band, activated_id | one_hot → `OneHotFeature` | `nn.Embedding` 조회 | 2씩 |
| slot | one_hot → `OneHotFeature` | 임베딩 | 8 |
| click_image_hashes | sequence → `SequenceFeature` | 최근 최대 100개 임베딩의 **합** | 30 |
| iab_category_history | sequence → `SequenceFeature` | 임베딩 합 | 10 |
| ufos | sparse_dense → `SparseDenseFeature` | 800차원 희소 벡터 → Linear | 10 |
| other_dsp_click_history | sparse_dense | 32차원 → Linear | 4 |
| click_count_history | dense → `DenseFeature` | 128차원 → tanh → Linear | 8 |
| n_other_dsp | dense | 값 1개 → Linear | 1 |
| **ecpm** | dense | **log1p(ecpm) → Linear(1→1)** | 1 |
| (network·service 등) time | `TimeFeature` | 주·일 주기를 sin/cos 4개로 | 4 |

- 모든 출력을 이어 붙이면(concat) talk 기준 **78차원** 벡터가 돼요.
- 임베딩 테이블 크기는 인덱스 수(`n_index`) × 1.2로 여유를 둬요.
- 학습되지 않는 auxiliary 피처
  - 수수료(`agency/dsp_commission_ratio`)와 `bidfloor`: 서빙 계산식에만 써요.
  - `pctr`, `pcvr`, `win`: 평가 지표에만 써요.

## 3. 네트워크: `DCN_v2.forward`

```python
x0 = concat(피처 출력)                  # [batch, 78]

# ① cross layer (crosslayers: [same] → 1층, 78→78)
x = x0 * Linear(x0) + x0               # x0 ⊙ (W·x + b) + x  ← 피처 간 곱(교차) 학습
x = ReLU(x)

# ② dense layer (layers: [32])
x = ReLU(LayerNorm(Linear(78→32)(x)))

# ③ 출력
out = Linear(32→1)(x)                  # 스칼라

# WpCrossv2.forward_network
bid_price = exp(out)                   # 항상 양수인 가격
```

- **cross layer**는 "입력 × 입력" 형태의 교차항을 명시적으로 만들어요. 예를 들어 "slot × ecpm"이나 "유저 × n_other_dsp" 같은 상호작용을 MLP보다 적은 파라미터로 배워요.
- 마지막에 **exp**를 쓰므로 네트워크는 사실상 **log 가격**을 출력해요. 이 값은 ecpm과 아무 제약으로도 묶여 있지 않아요. ecpm은 입력 78차원 중 1차원일 뿐이에요.
- 모델 크기가 작아요(cross 1층 + dense 32). 10분 안에 학습을 끝내야 하는 제약 때문이에요. 위키 과제 ⑪(학습 주기 재검토)이 이 제약을 다뤄요.

## 4. 학습

### 데이터 (`SequoiaDataloader`)

- 세콰이어 파이프라인 `winner-talk-all-bid-serving-join`의 10분 단위 파일을 읽어요.
- 모델 단계 필터
  - `ecpm >= 75` (지면별 하한)
  - 타겟이 비어 있지 않을 것 (`ssp_close_price > 0` 또는 win)
  - `n_bid_dsp == 0` 제외 (외부 DSP가 입찰한 경매만)

### loss (`WpWrapModel.calc_loss` → `log_smooth_l1_loss`)

```python
y_hat = log1p(bid_price)               # 예측
y     = log1p(ssp_close_price)         # 정답: 다른 플랫폼 최고가
loss  = smooth_l1(y_hat, y) * weights  # 합계(sum)
```

- 오차가 1(로그 공간에서 약 2.7배 차이) 이내면 제곱 오차, 그 밖이면 절대 오차예요. 그래서 크게 틀린 샘플의 영향이 제한돼요.
- 선택 옵션
  - `weight` + `n_percent`: 배치 안에서 타겟 상위 n%에 가중치를 줘요.
  - `underbid_penalty`: 진 경매인데 예측이 정답보다 낮으면 가중치를 줘요.
  - talk config는 둘 다 쓰지 않고, daum config는 둘 다 써요.

### 옵티마이저와 학습 루프 (`run_online.train_model`)

- Adam(lr 0.001), batch 16,384, epoch 1
- 매 실행마다 이렇게 동작해요.
  1. 직전 모델(`inherit_model` 또는 자기 자신의 최신 버전)을 Tenth2에서 내려받아 가중치와 Adam 상태를 주입해요(`load_prev_model` → `inject_old_to_new`).
  2. `forget.step: 0.999`로 이어받은 Adam step 수를 살짝 줄여요.
  3. 새 10분치 파일마다 `trainer.test` → `trainer.fit` → (옵션) `test`를 돌려요.
- 안전장치
  - 매 step 전 gradient norm을 기록하고, NaN/Inf면 즉시 중단해요(`on_before_optimizer_step`).
  - export 직전에 가중치가 유한한지 다시 확인해요(`check_weights_finite`).
- 평가 지표는 `EfficientPriceMetrics`로 계산해 MySQL에 기록해요. 학습한 모델은 Tenth2에 올리고, 모델 메타는 redicoke에 갱신해요(`update_model`).

## 5. 서빙: ONNX 안에서 일어나는 일

`export_model`이 `forward_train`을 ONNX로 내보내요. export 중에만 실행되는 분기가 있어서, 서빙 그래프는 학습 때와 조금 달라요.

```python
r = 1 - ca/100 - cd/100
ecpm = ecpm * r                        # ① 수수료 반영 (eCPM')
bid_price = exp(DCN(features))         # ② 위와 같은 네트워크
dr  = max(bid_price, bidfloor*(1+2eps)) / ecpm
bsr = r * (1 - dr)
pwr = 1
return pwr, dr, bsr, bid_price
```

### "export 중에만 실행되는 분기"의 동작 원리

코드 곳곳에 `if torch.onnx.is_in_onnx_export():` 조건문이 있어요. 이 함수는 **`torch.onnx.export()`로 모델을 ONNX 파일로 변환하는 동안에만 True**예요.

ONNX 변환은 코드를 한 번 실행하면서 지나간 연산을 그래프로 기록하는 방식이에요. 그래서 이 조건문은 이렇게 나뉘어요.

- 학습(`trainer.fit`) 중에는 **False** → 해당 코드가 실행되지 않아요.
- ONNX 변환 중에는 **True** → 해당 코드가 **서빙 그래프에 박혀요.**

결과적으로 **같은 Python 코드에서 학습 그래프와 서빙 그래프가 다르게** 만들어져요.

### 분기 4곳

| # | 위치 | 학습 때 | 서빙(ONNX) 때 | 이유 |
|---|---|---|---|---|
| ① | `forward_train` 맨 앞 | 없음 | `ecpm = ecpm × (1 − ca − cd)` | 수수료를 뺀 금액 기준으로 계산하려고 |
| ② | `forward_network` | 모든 피처가 batch N | 유저 피처를 광고 피처의 행 수만큼 복제(`expand`) | torch-dnn 공통 서빙 규약(유저 1벌 + 광고 N벌)을 물려받은 템플릿. 복제해도 값이 같아 **위너 예측 결과에는 영향 없음** |
| ③ | dr 계산 | `dr = bid_price / ecpm` | `dr = max(bid_price, bidfloor) / ecpm'` | bidfloor 아래로는 깎지 않으려고 |
| ④ | 출력 | `bsr = None`, `pwr = (bid_price ≥ 정답)` | `bsr = r × (1 − dr)`, `pwr = 1` | 학습 때는 정답과 비교한 승패, 서빙 때는 정답이 없으니 고정값 |

### ② expand는 torch-dnn 템플릿이에요

`expand(batch_size, …)` 코드는 위너만의 설계가 아니에요. CTR·CVR 모델(AdCross, DeepInterest, MaskNet 등)을 포함한 torch-dnn 모델 대부분에 똑같이 들어 있어요. 위너는 이 공통 서빙 규약을 그대로 물려받았어요.

| | CTR 모델 | 위너 |
|---|---|---|
| 서빙 입력 | 유저 1벌 + 광고 후보 **수백 벌** | 유저 1벌 + 광고 **1벌** (1등 광고, 추정) |
| 광고마다 다른 입력 | 소재·광고그룹·광고주 등 여러 개 | ecpm뿐 |
| expand의 의미 | **핵심 최적화**: 유저 피처를 1번만 계산해 수백 개 광고에 재사용 | 템플릿이라 있을 뿐. 광고가 1개면 복제할 것이 없고, N개여도 값이 같아 **결과에 영향 없음** |

- 학습 때는 행마다 (유저, 광고) 한 쌍이 들어 있어서 expand가 필요 없어요.
- 위너도 유저 피처를 많이 써요(talk 기준 입력 78차원 중 77차원). expand는 유저 피처를 **쓰는 방식**이 아니라 서빙 요청의 **모양을 맞추는 처리**일 뿐이에요.
- 경쟁 slot에서 실제로 광고가 몇 개 들어오는지는 a2 서빙 코드를 확인하지 않았어요. 역할상 1등 광고 1개로 추정해요.
- 비경쟁 slot의 SCM은 예외예요. 내부 2등 가격을 찾아야 해서 후보 여러 개를 함께 받아요.

### 주의할 점: ①은 결과를 바꿀 수 있어요

②·③·④는 서빙에 필요한 처리를 덧붙이는 것이라 모델이 배운 내용과 충돌하지 않아요. 하지만 **①은 네트워크 입력 자체를 바꿔요.**

- ① 이후의 `ecpm`이 dr의 분모뿐 아니라 **피처 `log1p(ecpm)`에도 그대로 들어가요.**
- 학습 때 ecpm 10,000 → 입력 `log1p(10,000) ≈ 9.21`
- 서빙 때 10,000 × 0.85 = 8,500 → 입력 `log1p(8,500) ≈ 9.05`

학습 데이터의 `ecpm` 컬럼이 이미 수수료를 뺀 값이라면 둘이 일치해서 문제가 없어요. 원래 eCPM이라면 **서빙 입력이 항상 낮게 들어가는 학습/서빙 불일치(skew)**가 생겨요.

위키 학습 데이터 표에는 as-is 입력이 "ecpm = sspPrice / dr", to-be가 "eCPM × (1 − ca − cd)"로 적혀 있어요. ADRECBID-2811이 바로 이 기준을 맞추려는 논의예요. 실제 컬럼 값이 어느 쪽인지는 데이터로 확인해야 해요.

참고로 학습 중 기록되는 dr 지표도 ①이 빠진 `bid_price / 원래 ecpm`이라서, 서빙 dr과 기준이 달라요.

## 6. 정리

- 위너가 bid_price를 예측하는 방식은 이래요. **유저·지면·광고 피처를 임베딩해 78차원으로 이어 붙이고, DCN v2(교차 1층 + Dense 32)를 거쳐 나온 스칼라에 exp를 씌워요.**
- 정답은 **다른 플랫폼 최고가**, loss는 **로그 공간 smooth L1**이에요.
- **10분마다 직전 모델을 이어받아** 새 데이터로 1 epoch씩 학습해요.
