# Hardware Optimization

## Collective Operations
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
https://en.wikipedia.org/wiki/Collective_operation

## AllReduce
https://medium.com/@niruthiha2000/allreduce-explained-the-key-to-efficient-distributed-training-2cbbcc871832

## Gradient Checkpointing

## FlashAttention-2
https://arxiv.org/pdf/2307.08691

## Mixed Precision
bf16, fp16, ...
https://bo-10000.tistory.com/32

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