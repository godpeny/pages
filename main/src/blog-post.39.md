# Hardware Optimization

## Collective Operations
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
https://en.wikipedia.org/wiki/Collective_operation

## AllReduce
https://medium.com/@niruthiha2000/allreduce-explained-the-key-to-efficient-distributed-training-2cbbcc871832

## Gradient Checkpointing

## FlashAttention-2
https://arxiv.org/pdf/2307.08691


## Precision
컴퓨터가 숫자를 표현할 때는 0과 1(비트)을 사용합니다. 비트를 많이 쓸수록 숫자를 더 정밀하게 표현할 수 있지만, 그만큼 메모리를 많이 차지하고 계산 속도가 느려집니다.
- 단정밀도 (FP32 / Single Precision): 숫자를 표현하는 데 32비트를 씁니다. 소수점 아래아주 작은 수까지 정확하게 표현할 수 있어서 기본 정밀도로 오랫동안 사용되어 왔습니다.
- 반정밀도 (FP16 / Half Precision): 숫자를 표현하는 데 16비트만 씁니다. FP32에 비해 메모리를 딱 절반만 차지하고 계산 속도가 훨씬 빠르지만, 표현할 수 있는 숫자의 범위가 좁고 정밀도가 떨어집니다.

### Mixed Precision
Single Precision과 Half Precision 두 개의 장점만 섞어서 사용하는 방식입니다.
덧셈, 곱셈이 수억 번씩 일어나서  시간이 오래 걸리는 거대 연산은 가볍고 빠른 FP16 복사본으로 하고 연산 결과로 나온 미세한 수정치들을 안전하게 누적 기록해야 하는 원본 저장은 FP32로 하는 것입니다. 32비트 원본을 16비트로 변환하는 과정이 추가되지만, 이 변환 속도보다 16비트로 수억 번의 행렬 계산을 아끼는 속도 이득이 압도적으로 크기 때문에 이 방식이 가능하고 또 유용한 것입니다.

엔비디아(NVIDIA) 등에서는 안정적인 학습을 위해 다음과 같은 장치들을 만들었습니다.  

<b> FP32 마스터 가중치(Master Weights) 유지 </b>  
학습 중에는 아주 미세한 변화(경사 하강법에 의한 가중치 업데이트)를 기록해야 합니다. FP16은 너무 미세한 숫자를 표현하지 못하므로, 가중치의 '원본'은 FP32로 따로 저장해 둡니다. 계산할 때만 FP16 복사본을 만들어 쓰고, 최종 업데이트는 원본(FP32)에 적용하는 방식입니다. 즉, 32비트에서 16비트만 잘라내서 계산하고, 다시 32비트로 만들어 ($0.00123$ -> $0.00123000000000...$) 업데이트 합니다. 

| 학습 회차 (Batch)       | 계산된 FP16 미분값 | FP32 변환 값 | 가중치 업데이트 연산       | 최종 업데이트된 FP32 원본 값 |
| ------------------- | -----------: | --------: | ----------------- | -----------------: |
| 시작 전                |            - |         - | -                 |            0.54321 |
| 1번째 배치              |        -0.05 |     -0.05 | 0.54321 - 0.05000 |            0.49321 |
| 2번째 배치              |        -0.05 |     -0.05 | 0.49321 - 0.05000 |            0.44321 |
| 3번째 배치              |        -0.05 |     -0.05 | 0.44321 - 0.05000 |            0.39321 |
| 4번째 배치              |        -0.09 |     -0.09 | 0.39321 - 0.09000 |            0.30321 |
| 5번째 배치              |        -0.01 |     -0.01 | 0.30321 - 0.01000 |            0.29321 |
| 6번째 배치              |        -0.09 |     -0.09 | 0.29321 - 0.09000 |            0.20321 |
| 🔥 7번째 배치           |        -0.01 |     -0.01 | 0.20321 - 0.01000 |            0.19321 |
| 🔥 8번째 배치           |        -0.10 |     -0.10 | 0.19321 - 0.10000 |            0.09321 |
| 💥 9번째 배치 (결정적 순간)  |        -0.09 |     -0.09 | 0.09321 - 0.09000 |            0.00321 |
| 🚀 10번째 배치 (뒷자리 붕괴) |        -0.01 |     -0.01 | 0.00321 - 0.01000 |           -0.00679 |

위 예시처럼 매 순간순간 나오는 미분값의 해상도는 16비트 수준이 맞지만, 서로 다른 16비트 화살표들을 32비트 공간에 촘촘하게 누적 시켜 그리다 보면 결국 정밀하고 매끄러운 32비트짜리 완성형 곡선(최적의 가중치)이 만들어지는 원리입니다.


<b> 손실 스케일링 (Loss Scaling) </b>  
딥러닝 학습 시 미분 값(Gradients)은 0에 가까운 아주 작은 소수일 때가 많습니다. FP16은 너무 작은 숫자를 표현하지 못해 이 값들을 그냥 0으로 처리해 버리는 언더플로우(Underflow) 현상이 발생합니다.
이를 막기 위해, 손실(Loss) 값에 특정 수(예: 8, 32000 등)를 곱해서 숫자를 일시적으로 뻥튀기(Scaling)해 줍니다. 계산이 끝난 후 다시 그 수만큼 나눠서 원래 크기로 돌려놓는 기법입니다.

<b> 텐서 코어(Tensor Core) 활용 </b>  
최신 NVIDIA GPU(Volta, Turing, Ampere architecture 등)에는 텐서 코어라는 특별한 하드웨어가 탑재되어 있습니다. 이 친구는 FP16 연산을 하드웨어 차원에서 무지막지하게 빠른 속도로 처리(FP16으로 곱하고, 결과는 FP32로 안전하게 더함)해 주기 때문에 혼합 정밀도를 쓸 때 극적인 속도 향상을 얻을 수 있습니다.

https://bo-10000.tistory.com/32
https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html

## DeepSpeed
부ㄴ산 학습 메모리를 효율적이고 빠르게 만드는 PyTorch 최적화 라이브러리입니다. 그 핵심은 대규모 모델을 규모에 맞게 훈련할 수 있는 Zero Redundancy Optimizer(ZeRO)입니다.
### Zero Redundancy Optimizer(ZeRO)
DeepSpeed의 핵심 기능이다. 파라미터, 그라디언트, 옵티마이저 상태를 GPU끼리 분산(shard)해 GPU 메모리 사용량을 대폭 절감한다. 예를 들어 ZeRO-3는 파라미터 자체까지 shard하여 GPU 하나가 모델 전체를 항상 올리지 않고도 학습이 가능하다. PyTorch DDP는 모델 전체를 각 GPU에 복제하는 방식이라 메모리 사용량이 높지만, ZeRO는 이를 최소화한다.

ZeRO-1: optimizer states 분산
ZeRO-2: optimizer states + gradients 분산
ZeRO-3: optimizer states + gradients + parameters 까지 모두 shard

https://huggingface.co/papers/1910.02054

### PyTorch DDP
DistributedDataParallel (DDP) is a powerful module in PyTorch that allows you to parallelize your model across multiple machines, making it perfect for large-scale deep learning applications. To use DDP, you’ll need to spawn multiple processes and create a single instance of DDP per process.

But how does it work? DDP uses collective communications from the torch.distributed package to synchronize gradients and buffers across all processes. This means that each process will have its own copy of the model, but they’ll all work together to train the model as if it were on a single machine.

To make this happen, DDP registers an autograd hook for each parameter in the model. When the backward pass is run, this hook fires and triggers gradient synchronization across all processes. This ensures that each process has the same gradients, which are then used to update the model.

https://tutorials.pytorch.kr/beginner/dist_overview.html
https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html


## vLLM(PagedAttention)
https://arxiv.org/abs/2309.06180
https://pangyoalto.com/pagedattetion-review/
https://velog.io/@kaiba0514/vLLM%EC%9D%80-%EC%99%9C-%EB%B9%A0%EB%A5%B8%EA%B0%80-Paged-Attention