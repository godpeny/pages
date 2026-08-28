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
