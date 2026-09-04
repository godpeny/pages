# models/adwrapmodel.py
신경망을 학습시키는 "감독관"입니다. 실제 레이어(nn.Linear, 임베딩, Cross layer)는 이 파일에 단 한 줄도 없습니다. 대신 "그 네트워크를 어떻게 학습시키고, 어떻게 채점하고, 어떤 옵티마이저로 돌릴지"만 정의합니다.

## torch-dnn의 3계층 구조에서의 위치
```
┌────────────────────────────────────────────────────────┐
│ [1] Trainer          pytorch_lightning.Trainer         │
│     학습 루프, GPU 배치, 체크포인트 — 라이브러리 제공            │
└───────────────────────┬────────────────────────────────┘
                        │ 정해진 훅(hook)을 호출
                        ▼
┌────────────────────────────────────────────────────────┐
│ [2] Wrapper          ★ adwrapmodel.py (이 파일) ★        │
│     • 배치에서 정답 라벨 분리                                │
│     • loss 계산                                         │
│     • 광고 도메인 메트릭 계산 (RIG, calibration, AUC)        │
│     • 네거티브 샘플링 역보정                                 │
│     • 옵티마이저 구성                                       │
│     ※ 레이어 정의는 없음                                    │
└───────────────────────┬────────────────────────────────┘
                        │ self.main_net.forward_train(x)
                        ▼
┌────────────────────────────────────────────────────────┐
│ [3] Network          models/model/adsimple.py 등        │
│     임베딩 + Dense/Cross 레이어 + Sigmoid                  │
│     실제 파라미터가 사는 곳                                  │
└────────────────────────────────────────────────────────┘
```
### Wrapper 역할
<b> Lightning이 요구하는 규약을 채워야 합니다. </b>  
``Trainer.fit(model, datamodule)`` 한 줄로 학습이 돌아가는 대신, Lightning은 모델이 정해진 메서드를 구현하고 있을 것을 요구합니다. Wrapper 파일의 거의 전부가 이 Lightning이 호출하는 훅의 목록을 채우는 코드입니다.

- training_step
- validation_step
- test_step
- on_validation_epoch_end
- on_test_epoch_end
- configure_optimizers	

<b> 네트워크와 학습 방식을 직교(orthogonal)하게 조합하기 위함입니다. </b>
AdSimple, AdCrossv2, DeepInterest, MaskNet 등 torch-dnn의 주요 모델들은 전부 "피처 → 확률" 함수라는 점에서 동일합니다. 그러면 loss 계산·메트릭·옵티마이저 코드를 네트워크마다 매번 복사할 이유가 없습니다.

## __init__
YAML config 하나를 받아 학습 가능한 객체를 조립합니다.
```python
def __init__(self, config):
        super().__init__()
        self.model_config = config.model
        self.config = config.hyper_parameters
        self.target = [f["name"] for f in config.model.features if f["feature_type"] == "target"][0]

        # Define model
        self.main_net = getattr(model, config.model.mainFF.get("type", "AdSimple"))(config)
        self.negative_sample_ratio = self.main_net.negative_sample_ratio

        # Define loss func
        self.loss_func_eval = nn.BCELoss(reduction="sum")

        if "loss_info" in config.hyper_parameters:
            self.loss_func_train = get_loss_type(config.hyper_parameters.get("loss_info"))
        else:
            # default
            self.loss_func_train = self.loss_func_eval

        self.metrics = {"auc": torchmetrics.AUROC(task="binary")}
        self.save_hyperparameters()

        self.validation_step_outputs = []
```
```yaml
# config/ctr/network/ctr_online_network_v9.107.3.yaml (구조 예시)
name: ctr_online_network_v9.107.3
online:
  negative_sample_ratio: 0.1          # 네거티브 10%만 사용
model:
  features:
    - name: gender
      feature_type: one_hot_feature
      output_dim: 2
    - name: slot
      feature_type: one_hot_feature
      output_dim: 12
    - name: click                     # ← 이게 타겟
      feature_type: target
  mainFF:
    type: AdCrossv2
    layers: [96, 48]
hyper_parameters:
  loss_info:
    type: smoothing
    alpha: 0.1
  optimizer:
    name: adam
    learning_rate: 0.001
```
이 config로 ``_init`` 을 통해 AdWrapModel(config)를 만들면 아래와 같습니다.

- model.model_config: {features: [...], mainFF: {...}}
- model.config: {loss_info: {...}, optimizer: {...}}  (hyper_parameters 서브트리)
- model.target: "click"
- model.main_net: AdCrossv2 인스턴스
- model.negative_sample_ratio: 0.1
- model.loss_func_train: smoothing 클로저 (alpha=0.1)
- model.loss_func_eval: BCELoss(reduction='sum')
- model.metrics["auc"]: AUROC 객체 (비어 있음)
- model.validation_step_outputs: []

## forward / forward_recalibration
- forward: 학습 시 사용, sigmoid 적용된 예측값 반환.
- forward_recalibration: $p/(1-p)$는 CVR의 fake negative calibration 보정 공식 적용 한 것.

## training_step & loss
```python
def training_step(self, batch, batch_idx):
  y = batch.pop(self.target)
  x = batch
  y_hat = self.forward(x)

  loss, metrics = self.calc_loss(y_hat, y)

  ret_log = {"loss": loss}
  ret_log.update(metrics)

  self.log_dict(ret_log)
  return ret_log

.
.
.

def calc_loss(self, y_hat, y):
  if y_hat.shape[1] == 2:
    y = torch.stack([y, 1.0 - y], dim=1)

  if y.dim() != y_hat.dim():
    ce_loss = self.loss_func_train(y_hat[:, 0], y)
  else:
    ce_loss = self.loss_func_train(y_hat, y)

  with torch.no_grad():
    metrics = {}
    metrics["clk_sum"] = calc_clk_sum(y)
    metrics["pctr_sum"] = calc_pctr_sum(y_hat)
    metrics["rig"] = calc_rig(y_hat, y, y.shape[0], ce_loss)
    
    return ce_loss, metrics

.
.
.

# Define loss func
self.loss_func_eval = nn.BCELoss(reduction="sum")

if "loss_info" in config.hyper_parameters:
  self.loss_func_train = get_loss_type(config.hyper_parameters.get("loss_info"))
else:
  # default
  self.loss_func_train = self.loss_func_eval
```
batch에서 레이블(y)과 피쳐(x)를 분리 한 후, 앞서 ``forward``에서 등록한 메서드를 호출, 예측값 y_hat을 도출합니다.  
calc_loss 에서는 y와 y_hat을 입력 받아 미리 정의한 loss function으로 오차를 구합니다.  
loss와 metric을 dictionary에 넣은 후 lightning 에 로깅정보로 전달합니다.


## validation
```python
def validation_step(self, batch, batch_idx):
        y = batch.pop(self.target)
        x = batch
        y_hat = self.forward_recalibration(x) if self.config.get("recalibration", False) else self.forward(x)

        metric = self.eval_metrics(y_hat, y)

        self.validation_step_outputs.append(metric)
      
        return metric
.
.
.

def eval_metrics(self, y_hat, y):
  metrics = {}
  # low level inform.
  metrics["clk_sum"] = calc_clk_sum(y)
  metrics["pctr_sum"] = calc_pctr_sum(y_hat)
  metrics["cal"] = metrics["pctr_sum"] / metrics["clk_sum"]
  metrics["imp_sum"] = (y_hat.shape[0] - metrics["clk_sum"]) * 1 / senegative_sample_ratio + metrics["clk_sum"]
  metrics["true_positive"] = calc_true_positive(y, y_hat)

  self.metrics["auc"].update(y_hat[:, 0], y.int())

  # high level inform.
  cal_y_hat = (
    y_hat[:, 0] * self.negative_sample_ratio / (y_hat[:, 0] * snegative_sample_ratio + (1 - y_hat[:, 0]))
  )

  metrics["loss_sum"] = log_loss(cal_y_hat, y, self.negative_sample_ratio)
  return metrics
```

validation_step 에서는 train과 거의 같지만 train에서는 ``calc_loss``로 오차를 측정해 backpropagation에서 필요한 gradient를 구하려고 했다면, 평가 단계에서는 gradient가 필요 없고, 대신 네거티브 샘플링 역보정이 들어간 지표를 도출하는 ``eval_metrics``을 호출합니다.  
평가 지표는 전체 데이터를 합산해야 의미가 있습니다(캘리브레이션, RIG, AUC 모두). 그래서 배치별 부분합을 리스트에 쌓아두고 에포크 끝에서 처리합니다.  

### 원분포 복원
#### 원래 노출 복원
```python
metrics["imp_sum"] = 
(y_hat.shape[0] - metrics["clk_sum"]) * 1 / senegative_sample_ratio 
+ metrics["clk_sum"] 
```
- 배경: 네거티브 9,900 + 클릭 100  =  총 노출 10,000
- 샘플링 비율: 네거티브만 10% 샘플링 (클릭은 전부 유지)
- 샘플: 네거티브 990 + 클릭 100  =   1,090 (N = ``y_hat.shape[0]``)

```python
imp_sum = (N - clk_sum) / ratio + clk_sum
        = (1090 - 100) / 0.1 + 100
        = 990 / 0.1 + 100
        = 9900 + 100
        = 10000 
```

평가 단계에서 Negative Sampiling 을 쓰지 않고 원 분포를 복원하는 이유? 네거티브를 10%만 남기고 학습했다면(ratio = 0.1), 부풀려진 수치이기 때문입니다.
```
실제: 노출 10,000건, 클릭 100건    → CTR = 1.0%
학습 데이터: 네거티브 9,900 × 0.1 = 990건 + 클릭 100건 = 1,090건
           → 관측 CTR = 100/1,090 = 9.2%      ← 약 9배 부풀려짐
```
Negative Sampling 때문에 모델은 9.2%를 출력하도록 학습됩니다. 그런데 입찰에 필요한 건 1.0%입니다. 평가 지표는 실제 좌표계에서 내야 합니다.

### 원래 노출 공간의 CTR 복원
모델이 학습한 공간: 네거티브를 r배로 줄인 샘플링 공간 -> 여기서의 CTR을 p_s라 하면 실제보다 부풀려져 있음.  
평가에 필요한 값: 샘플링 없는 원래 노출 공간의 CTR
```python
 cal_y_hat = (
    y_hat[:, 0] * self.negative_sample_ratio / (y_hat[:, 0] * snegative_sample_ratio + (1 - y_hat[:, 0]))
  )
```

```python
q = p / (p + (1-p)·r) <=> q·(p + (1-p)r) = p
qp + qr - qrp   = p
qr              = p - qp + qrp
qr              = p·(1 - q + qr)

           qr
p = ─────────────────
     q·r + (1 - q)

cal_y_hat = y_hat * r / (y_hat * r + (1 - y_hat))
#   ^       ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
#   p          q·r          q·r + (1-q)     
```

- ``cal_y_hat``: 예측 CTR 을 원래의 공간으로 복원해 이후 계산할 loss 값이 원래의 공간 기준으로 계산되게 합니다. 
- ``log_loss(..., self.negative_sample_ratio)``: 네거티브 행 1건을 원공간 1/r건으로 센 후 ``cal_y_hat`` 를 곱함

즉, ``cal_y_hat``는 "얼마나 틀렸나"를, ``log_loss(..., self.negative_sample_ratio)`` 는 "그게 몇 건에 해당하나"를 고친후 ``cal_y_hat`` 와 곱해 loss_sum을 구합니다.

## on_validation_epoch_end
- ``val_log_loss = loss_sum  /  ((N − clk)/r + clk)``: 분자(loss_sum)에서 네거티브 항을 1/negative_sample_ratio 로 불렸으니, 분모도 원공간 노출 수로 마찬가지로 네거티브를 보정합니다. 

### RIG (Relative Information Gain)
```python
ctr     = clip(val_clk_sum / val_imp_sum, 1e-6, 1-1e-6)     # 실제 CTR
entropy = -(ctr·log(ctr) + (1-ctr)·log(1-ctr))               # 상수 예측기의 log loss
val_rig = 1 - val_log_loss / entropy
```
entropy는 항상 CTR만 출력하는 모델의 log loss입니다.  
이 entropy를 baseline으로 구한 loss를 정규화 해서 CTR을 상수로만 예측하는 최선의 무지 모델 대비 몇 % 개선했나를 나타내는 지표입니다.

## configure_optimizers
온라인 증분학습이 이루어지는 부분입니다. Lightning은 fit() 시작 시 configure_optimizers()를 호출해 옵티마이저를 만듭니다. 이 프로젝트의 온라인 학습은 매 사이클마다 이전 모델을 이어서 학습합니다.
```python
    def configure_optimizers(self):
        if self.trainer.optimizers:
            return self.trainer.optimizers
        else:
            return self.build_optimizers()
```
``run_online_cvr.py`` 의 흐름은 이렇습니다.
```python
DummyDataloader(config).warmup_trainer(model, trainer)    # trainer.optimizers 슬롯 생성
load_prev_model(...)  →  inject_old_to_new(...)           # 이전 옵티마이저 상태 주입
trainer.fit(model, datamodule=data_loader)                # ← configure_optimizers 호출
    → self.trainer.optimizers 가 이미 채워져 있음
    → 그대로 반환 → 주입된 모먼트 보존 ✓
```
아래 예시를 참고합니다.

```yaml
# ── 최초 학습 (2026-08-21 00:00, 이전 모델 없음) ──
self.trainer.optimizers == []  # -> falsy
build_optimizers() # -> Adam(lr=0.001), exp_avg = 0
# 학습 후 Tenth2 업로드

# ── 증분 학습 (2026-08-21 01:00) ──
warmup_trainer() # trainer.optimizers = [Adam(...)]  (빈 상태)
load_prev_model() # Tenth2에서 00:00 모델 다운로드
inject_old_to_new() # -> 가중치 + exp_avg / exp_avg_sq 주입
trainer.fit()
#      → configure_optimizers()
#      → self.trainer.optimizers 가 truthy → 그대로 반환
#      → 00:00 시점의 Adam 모먼트 유지 ✓  (learning rate warm-up 불필요)
```

# models/addfmmodel.py
| 모델 | AdWrapModel | AdDFMModel |
|---|---|---|
| 배치에서 꺼내는 것 | `y` | `y` + `delay`(정규화·클리핑) |
| forward 반환 | 텐서 1개 | **튜플 (pCVR, λ)** |
| loss 호출 | `loss(ŷ, y)` | `loss(ŷ, λ̂, y, d)` |

- λ: 지수분포의 rate 파라미터로 "전환한다면 평균 1/λ일 후에"를 의미합니다.
- d: 딜레이
  - y=1: 클릭→전환에 실제로 걸린 시간, loss에서 λ가 이 시간 분포를 배우게 함.
  - y=0: 클릭 후 지금까지 기다린 시간

d는 관측값, λ는 그 d가 따르는 분포의 파라미터입니다. DFM은 "전환이 일어난다면, 클릭 후 걸리는 시간 d는 지수분포(Exp(λ))를 따른다"고 가정합니다.

# models/addfmmodel.py
| 모델 | AdWrapModel | AdDFMModel | **AdDFOMModel** |
|---|---|---|---|
| 배치에서 pop | `y` | `y` + `delay` | **`y`만** (delay 개념이 래퍼에 없음) |
| forward 반환 | 텐서 1개 | 튜플 (p, λ) | **텐서 1개** |
| loss 호출 | `loss(ŷ, y)` | `loss(ŷ, λ̂, y, d)` | **`loss(ŷ, y)`** — fnw도 2-인자 |

ADDFOMMOdel은 AdWrapModel와 거의 동일합니다. AdWrapModel에 `loss_info: {type: fnw}`를 넣어도 학습은 돌아갑니다. 이 파일이 별도로 존재하는 이유는 학습이 아니라 평가 좌표계(`eval_metrics_dfom`) 때문입니다.

## eval_metrics_dfom
```yaml
# 클릭 1,000건 (진짜 CVR 5%) 의 스트림
y=0 행:  1,000건   <- 클릭당 정확히 1건
y=1 행:     50건   <- 전환된 클릭의 재등장 (y=0 와 중복)
합계:    1,050건
```
일반 eval_metrics처럼 pctr_sum = 전체 행의 Σŷ로 하면, 전환된 클릭 50건은 두 번 세어집니다 (y=0으로 한 번, y=1로 한 번). 분자에 중복이 끼면 cal이 체계적으로 부풀어 캘리브레이션 판정이 불가능해집니다.

```python
    def eval_metrics_dfom(self, y_hat, y):
        metrics = {}
        # conv_data = y.bool()
        click_data = y.logical_not()
        cal_y_hat = (
            y_hat[:, 0] * self.negative_sample_ratio / (y_hat[:, 0] * self.negative_sample_ratio + (1 - y_hat[:, 0]))
        )
        # low level inform.
        metrics["clk_sum"] = calc_clk_sum(y)
        metrics["pctr_sum"] = calc_pctr_sum(cal_y_hat[click_data]) * 1 / self.negative_sample_ratio
        metrics["cal"] = metrics["pctr_sum"] / metrics["clk_sum"]
        metrics["imp_sum"] = (y_hat.shape[0] - metrics["clk_sum"]) * 1 / self.negative_sample_ratio
        metrics["true_positive"] = calc_true_positive(y, y_hat)

        self.metrics["auc"].update(y_hat[:, 0], y.int())

        # high level inform.
        metrics["loss_sum"] = log_loss(cal_y_hat, y, self.negative_sample_ratio)

        return metrics
```
``click_data = y.logical_not()`` y=0인 행만 골라내는 마스크로 DFOM에서 y=0 행은 "클릭당 정확히 1건"이므로, 곧 중복 없는 클릭 모집단을 의미합니다.  
마찬가지로 ``imp_sum = (N - clk_sum)/r`` 에서 ``models/adwrapmodel`` 과 달리 + clk_sum이 없습니다.

```
일반(노출 좌표계):  imp_sum = 네거티브복원 + 클릭 -> 분모 = 노출 수
DFOM(클릭 좌표계):  imp_sum = y=0 행 수 = 클릭 수  -> 분모 = 클릭 수 (y=1 행은 중복이므로 모집단에 더하면 안 됨)
```

# models/modules/loss.py
## mtldfm_v2
```python
def mtldfm_v2(y_hat, delay_hat, y_positive, delay_positive, y_negative, delay_negative, target_type):
    """DFM v2 loss with negative mask support."""
    target_type_ = target_type.view(-1, 1).long()
    y_ = y_positive.unsqueeze(1)
    delay_ = delay_positive.unsqueeze(1)
    delay_negative_ = delay_negative.unsqueeze(1)

    delay_hat_ = delay_hat.gather(1, target_type_)      # 해당 head의 λ만 추출
    y_hat_ = y_hat.gather(1, target_type_)              # 해당 head의 p만 추출

    # Negative samples loss — 모든 head에 적용, neg_conv로 마스킹
    output = (
        (1.0 - y_negative) * torch.log(1.0 - y_hat + y_hat * torch.exp(-delay_hat * delay_negative_) + FLOAT32_EPS)
    ).sum(1, keepdim=True)

    # Positive samples loss — target_type head 하나에만 적용
    output += y_ * (torch.log(y_hat_ + FLOAT32_EPS) + torch.log(delay_hat_ + FLOAT32_EPS) - delay_hat_ * delay_)

    return -torch.mean(output)
```
dfm loss(4-인자)를 K-head로 일반화한 것입니다. 

### 포지티브 항 — gather로 한 head만 조준
```python
y_hat_ = y_hat.gather(1, target_type_)     # [B,K]에서 [B,1]로: 샘플별 action_type
``` 
gather는 "샘플 b의 target_type[b]번째 열만 뽑아라"입니다. 전환이 관측된 샘플은 자신의 action_type head에만 긍정 신호를 더합니다. 즉, 구매 전환이 설치 head에 포지티브 신호를 주면 안 되도록 하는 장치입니다.

### 네거티브 항 — 모든 head에 브로드캐스트, 마스크로 구멍내기
```
(1.0 - y_negative) * log(1 - y_hat + y_hat·exp(-delay_hat·neg_delay) + EPS)
       └────┬────┘         └──────────────┬──────────────┘
    head별 제외 마스크      dfm의 생존혼합 네거티브 항을 [B,K] 전체에 적용
```
 K개 head 각각에 "neg_delay만큼 기다렸는데 이 종류의 전환은 안 왔다"는 네거티브 신호가 갑니다. 단 y_negative(=neg_conv)가 1인 head는 (1-1)=0으로 항(포지티브 항)이 소거됩니다. 전환이 관측된 head를 네거티브로 오염시키지 않기 위한 마스크입니다
# models/model/mtlcrossv2dfm.py
# models/model/mtlsimpledfom.py
# models/mtldfmmodel.py
AdDFMModel에 "태스크 축"을 추가한 멀티태스크 버전이자, 현재 프로덕션 DFM이 실제로 쓰는 래퍼입니다.  
AdDFMModel은 "전환이냐 아니냐" 하나만 예측하지만 실제 광고의 전환은 PURCHASE(구매), APP_INSTALL(앱 설치), SIGN_UP(가입), CART(장바구니) 등등 여러 목표를 가지고 있습니다.  

그러나 목적별로 모델을 따로 만들면 데이터가 쪼개져 소수 목적 모델은 늘 데이터가 부족합니다. 따라서 MTL(Multi-Task Learning)의 해법은 임베딩과 표현(representation)은 모델들이 전부 공유하고, 출력만 action_type별 K개 head로 분리하는 것입니다. 

예를 들면, 유저의 행동 패턴("이 유저는 광고에 반응하는 사람인가")은 목적이 달라도 공통 지식이므로 공유해서 배우고, "구매로 이어지나 vs 설치로 이어지나"의 차이만 head가 담당합니다. 소수 태스크가 다수 태스크의 데이터에 얹혀 학습되는 구조입니다.  


## 비교

DFM의 세 계약이 전부 **K차원으로 확장**됩니다:

| 모델 | AdDFMModel | **MTLDFMModel** |
|---|---|---|
| 배치에서 꺼내는 것 | `y` + `delay` | `y` + `delay` + **`neg_conv` + `neg_delay` + `action_type`** |
| forward 반환 | (p, λ) 각 `[B,1]` | (p, λ) 각 **`[B,K]`** — action_type별 head |
| loss 호출 | 4-인자 `dfm` | **7-인자 `mtldfm_v2`** |

- action_type: 전환 타입, 예) 1 = 구매, 2 = 설치, 3 = 가입
- neg_delay: 네거티브의 "기다린 시간", 예) 259200(초)) = 클릭 후 3일 경과
- neg_conv — "네거티브 취급 금지" 마스크:  예) [0, 1, 0, 0, 0] = "head 1은 네거티브로 세지 마라"

참고로 neg_delay가 재는 것은 "각 head가 기다린 시간"이 아니라 관측 창(observation window)의 길이입니다. 즉, ``neg_delay = 스냅샷 시각 − 클릭 시각`` 입니다.
```
클릭 (t=0) ────────────────────── 스냅샷 (t=3일)
   │                                  │
   ├─ 구매 head의 관측 시작: t=0      ├─ 구매 head의 관측 종료: t=3
   ├─ 설치 head의 관측 시작: t=0      ├─ 설치 head의 관측 종료: t=3
   └─ 카트 head의 관측 시작: t=0      └─ 카트 head의 관측 종료: t=3
```
즉 위와 같이 head가 같은 클릭에서 출발하고, 같은 스냅샷에서 멈춥니다. 이 스냅샷일때 전환이 된 것과 되지 않은 head의 구분은 neg_conv 를 통해서 적용합니다.  

```yaml
# 샘플: "클릭 3일 뒤 스냅샷. 2일째에 '구매' 전환이 있었다."

conv        = 1                      # "전환 있었음"
action_type = 1                      # "종류는 구매 → head 1로 배송"
delay       = 2일                    # "head 1의 포지티브 증거: 2일 걸림"
neg_conv    = [0, 1, 0, 0, 0]        # "head 1은 네거티브 계산에서 제외"
neg_delay   = 3일                    # "head 0,2,3,4의 네거티브 증거: 3일간 안 옴"

→ loss 조립 결과:
   head 1:        log(p₁) + log(λ₁) − λ₁·2       (포지티브)
   head 0,2,3,4:  log(1 − pₖ + pₖ·e^(−λₖ·3))     (네거티브)
   샘플 1건 → head 5개 전원 학습.
```

## Outputmask
학습이 끝난 모델은 클릭 하나에 대해 이렇게 답합니다.
```
모델 출력:  [ p구매=0.030,  p설치=0.008,  p가입=0.002,  p장바구니=0.019 ]
```
그런데 서빙(입찰) 쪽이 원하는 답은 아래와 같습니다.
```
입찰 계산:  bid = pCTR × pCVR × 광고주 입찰가      ← pCVR 자리에 숫자 "하나"
```
K개 중 어느 것을 쓸 것인가? 정답은 광고마다 다릅니다. 광고에는 캠페인 목적(objective_detail_type)이 붙어 있습니다.
```
광고 A: 목적 = PURCHASE      → 구매 확률이 pCVR
광고 B: 목적 = APP_INSTALL   → 설치 확률이 pCVR
```
즉 "K개 → 1개" 변환 규칙은 목적 → 유효 action_type 매핑이고, 이걸 행렬로 만든 것이 ``output_mask`` 입니다.


output_mask는 [목적 수 × K] 크기의 0/1 행렬입니다:
```
                 head:  pad  구매  설치  가입  장바구니
PURCHASE     행:      [  0,   1,   0,   0,   1  ]     ← 구매·장바구니가 유효
APP_INSTALL  행:      [  0,   0,   1,   0,   0  ]
SIGN_UP      행:      [  0,   0,   0,   1,   0  ]
```
서빙 forward의 실제 한 줄:
```python
output = (output * self.output_mask[inputs["objective_detail_type"].long()]).sum(1, keepdim=True)
```

```
입력 광고의 목적 = PURCHASE (idx 1)
① output_mask[1]        → [0, 1, 0, 0, 1]                      해당 행 꺼내기
② 출력과 원소별 곱      → [0, 0.030, 0, 0, 0.019]              무관한 head 소거
③ sum(1)               → 0.030 + 0.019 = 0.049              유효 head 합산
→ "이 광고 기준 pCVR = 4.9%"  ← 입찰에 들어가는 숫자 하나
```
유효 action이 하나뿐인 목적이면 단순 "선택"이고, 여럿이면(PURCHASE에 구매+장바구니) "합산"입니다. "그중 뭐라도 전환하면 전환"이라는 해석입니다.

``mtldfm_v2``loss 참고

# models/mtldfommodel.py

# config/cvr/dfm, config/cvr/dfom