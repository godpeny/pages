# Deep & Cross Network (DCN)
https://arxiv.org/abs/1708.05123

크로스 네트워크 (입력 초기값을 출력벡터와 계속 교차 + 이전 입력 잔차 더함)의 출력과 딥 네트워크 (일반적인 뉴럴네트워크)의 출력물을 concate 한 뒤 가중치를 곱하고 시그모이드를 통과 시켜서 확률을 얻는 구조.  

1. Cross Network
매 레이어마다 초기 입력값($x_0$)과 현재 레이어의 출력($x_l$)을 교차(cross)시키고, 여기에 이전 레이어의 입력값($x_l$)을 잔차(residual) 형태로 더해주는 구조입니다. 
$$x_{l+1} = x_0 x_l^T w_l + b_l + x_l$$

2. Deep Network
크로스 네트워크와 병렬로 작동하며, 일반적인 다층 퍼셉트론(MLP)처럼 활성화 함수(ReLU)를 거쳐 복잡한 비선형적 상호작용을 학습하는 일반적인 전방향 뉴럴 네트워크입니다.
$$h_{l+1} = ReLU(W_l h_l + b_l)$$

3. Combination Layer및 확률 도출**:
크로스 네트워크의 최종 출력($x_{L_1}$)과 딥 네트워크의 최종 출력($h_{L_2}$)을 하나의 벡터로 이어 붙입니다(concatenate). 그 후, 이 결합된 벡터에 최종 선형 가중치($w_{logits}$)를 곱한 뒤 시그모이드(Sigmoid, $\sigma$) 함수를 통과시켜 최종 예측 확률($p$)을 얻게 됩니다.
$$p = \sigma( [x_{L_1}^T, h_{L_2}^T] w_{logits} )$$

## DCN v2
https://arxiv.org/pdf/2008.13535

DCN v1은 Cross Network의 표현력(expressiveness)이 제한적이라는 단점이 있었는데, DCNv2는 수식 구조를 변경하여 표현력을 크게 높이면서도 연산 효율성을 유지하는 방향으로 발전했습니다. 핵심 수식들은 다음과 같이 전개됩니다.

### DCNv2 Standard Cross Layer
$$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$$
- $x_0$: 네트워크의 초기 입력 벡터 (임베딩 레이어의 출력).
- $x_l, x_{l+1}$: 각각 $l$번째와 $l+1$번째 크로스 레이어의 입출력 벡터.
- $W_l \in \mathbb{R}^{d \times d}$, $b_l \in \mathbb{R}^d$: $l$번째 레이어의 학습 가능한 가중치 행렬 및 편향.
- $\odot$: Hadamard product.

DCN v1이 입력 벡터와 스칼라 가중치를 곱하는 구조($x_0 x_l^T w_l$)여서 표현할 수 있는 다항식의 범위가 제한적이었던 반면, DCNv2는 가중치 행렬 $W_l$을 사용하여 훨씬 복잡하고 다양한 명시적 특징 교차(explicit feature crosses)를 학습할 수 있게 되었습니다.

### DCNv2 Cross Layer ver2: Low-Rank Cross Layer
$$x_{l+1} = x_0 \odot (U_l (V_l^T x_l) + b_l) + x_l$$
실제 산업 환경에서는 가중치 행렬 $W_l$의 크기($d \times d$)가 너무 커지면 연산량과 메모리 비용이 급증합니다. 이를 해결하기 위해 거대한 행렬 $W_l$을 두 개의 얇고 긴 행렬 $U_l$과 $V_l$로 분해(Low-Rank Approximation)**하는 수식이 도입되었습니다.  
여기서 $U_l, V_l \in \mathbb{R}^{d \times r}$ 이며, 랭크 $r$은 원래 차원 $d$보다 훨씬 작습니다 ($r \ll d$). 이 수식은 특징 교차 연산을 저차원 부분 공간(subspace)으로 투영했다가 다시 원래 차원으로 복원하는 방식으로 파라미터 수를 대폭 줄여줍니다.

### DCNv2 Cross Layer ver3: Mixture of Low-Rank Experts
$$x_{l+1} = \sum_{i=1}^K G_i(x_l) E_i(x_l) + x_l, \quad
E_i(x_l) = x_0 \odot (U_l^i (V_l^{iT} x_l) + b_l)$$
로우 랭크 구조의 성능을 극대화하기 위해, 단일 연산 대신 여러 개의 '전문가(Expert)' 네트워크를 두어 서로 다른 부분 공간에서 특징을 학습한 뒤 결합하는 방식(MoE; Mixture-of-Experts)을 제안했습니다.  
여기서 $K$는 전문가의 수, $G_i(x_l)$는 입력값에 따라 어떤 전문가의 결과에 더 가중치를 둘지 결정하는 게이팅 함수(Gating function, 주로 softmax 등 사용), $E_i(x_l)$는 $i$번째 전문가의 로우 랭크 크로스 연산 결과를 의미합니다.

#### DCNv2 Cross Layer ver4: Nonlinear Transformation
$$E_i(x_l) = x_0 \odot \left( U_l^i \cdot g(C_l^i \cdot g(V_l^{iT} x_l)) + b_l \right)$$
차원이 축소된 공간 안에서 표현력을 더 높이기 위해, 투영된 행렬 사이에 비선형 활성화 함수 $g(\cdot)$를 추가했습니다.

#### Deep and Cross Combination
이렇게 구성된 크로스 네트워크는 딥 네트워크(일반적인 ReLU 기반 다층 퍼셉트론)와 두 가지 구조로 결합될 수 있습니다.
- 직렬 연결 (Stacked Structure): $x_0$가 크로스 네트워크를 통과한 후 그 출력값($x_{L_c}$)이 딥 네트워크의 초기 입력값($h_0$)으로 들어가는 구조입니다.
- 병렬 연결 (Parallel Structure): $x_0$가 크로스 네트워크와 딥 네트워크에 동시에 입력된 후, 각 네트워크의 최종 출력($x_{L_c}$와 $h_{L_d}$)을 concate 하는 구조입니다. 

참고로 1번 수식이 DCNv2의 가장 기본이 되는 표준 원형(Standard Cross Layer)이며, 2, 3, 4번 수식은 실제 대규모 산업 환경의 자원 제약이나 성능 극대화 요구에 맞춰 고안된 상황별 변형(variation) 구조입니다.

최종적으로는 이 결합된 벡터에 선형 가중치를 곱하고 시그모이드(Sigmoid) 함수를 통과시켜 예측 확률을 도출하게 됩니다.