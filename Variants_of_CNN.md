2024년 09월 기준, 현대 딥러닝에서는 Attention이 가장 중요하다.  
`all you need is attention`은 2017년 6월 구글이 공개했으며 2024년 09월 29일 기준 134,394회 인용되었다.

[Attention is All You Need (Transformer)](https://velog.io/@tobigs-nlp/Attention-is-All-You-Need-Transformer)
- Transformer는 위 논문에서 등장한 모델로 기존의 seq2seq의 구조인 인코더, 디코더를 따르면서도, RNN을 사용하지 않고 Attention만으로 구현한 모델이다.  
- Transformer가 출현함으로써 BERT, GPT-3, T5 등과 같은 아키텍처가 발전하는 기반이 마련되었습니다.

* * *

1. SENet (CVPR 2018)
- https://deep-learning-study.tistory.com/539
- https://ffighting.net/deep-learning-paper-review/vision-model/senet/
- https://arxiv.org/abs/1709.01507
- https://github.com/hujie-frank/SENet

> CNN에서는 각 채널이 특정 정보를 담고 있지만 모든 정보가 풀고자 하는 문제와 관련이 있는 것은 아님.
> 따라서, 중요한 정보가 담긴 채널에만 집중하고 나머지는 무시하는 기능이 필요함.
> 하지만, Convolution, Activation, Pooling을 거치며 일부 중요한 정보가 더 두드러지긴 하지만 이를 명확하게 조정하는 것은 불가능함
> SE는 정보의 압축(Squeeze) 중요도계산 (Excitation)을 통해 성능을 향상

2. GoogleNet (CVPR 2014)
- https://arxiv.org/abs/1409.4842
- https://www.ytimes.co.kr/news/articleView.html?idxno=7149

3. Xception (CVPR 2017)
- https://arxiv.org/abs/1610.02357
- https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568

> (Original)Depthwise Separable Convolution
> - Step 1: Depthwise Convolution  
>   Convolution에서는 모든 입력 채널을 고려하여 하나의 필터를 적용하지만,  
>   Depthwise Convolution에서는 각 채널마다 **별도의 필터**를 사용.  
>   이로 인해 **채널 간의 상호작용은 고려되지 않고 각 채널은 독립적**으로 처리
> - Step 2: Pointwise Convolution  
>   Depthwise Convolution을 통해 각 채널별로 처리된 출력을 **다시 결합**하기 위해 1x1 Convolution을 적용  
>   이 과정에서 **채널 간의 상호작용을 고려**하며, Depthwise Convolution으로 축소된 연산 비용을 보충

> Xception에는 Step 2 이후 Step 1을 수행  
> 원래 Inception 모듈에서는 첫 번째 연산 후 비선형성이 있으나,  
> 수정된 깊이별 분리형 합성곱인 Xception에서는 중간 ReLU 비선형성이 없음

> 즉 Original CNN, SENet, (Original)Depthwise Separable Convolution, Xception으로 나뉘는 것.
> 이게 결국 attention 개념이다.

4. CBAM (ECCV 2018)
- arxiv.org/pdf/1807.06521.pdf
- https://blog.naver.com/winddori2002/222057978305
- https://ffighting.net/deep-learning-paper-review/vision-model/cbam/

> CBAM(Channel and Spatial Attention Module)은 CNN(Convolutional Neural Networks)에서 성능을 향상시키기 위해 제안된 어텐션 모듈  
> CBAM은 입력 피처 맵에서 중요한 정보를 강조하고 덜 중요한 정보를 억제함으로써, 네트워크가 더 중요한 특성에 집중할 수 있도록 도움  
> Channel Attention과 Spatial Attention의 두 가지 어텐션 메커니즘을 결합한 구조(위 3번까지는 Channel Attention만 구하던 것임)
```
Yes, you're correct in distinguishing the attention mechanisms used in **Xception** and **CBAM** (Convolutional Block Attention Module).

- **Xception**: While Xception (Extreme Inception) focuses on a deep convolutional architecture using depthwise separable convolutions, it does not explicitly implement attention mechanisms like channel or spatial attention as part of its core architecture. It uses efficient convolution techniques but does not use attention modules by default.

- **CBAM**: On the other hand, **CBAM** incorporates **both channel attention and spatial attention** mechanisms. Channel attention in CBAM focuses on the importance of different feature channels (which can be interpreted as the network's focus on "what" is important), while **spatial attention** identifies "where" the most relevant features are located in the image. So CBAM refines the feature representation both across channels and spatially.

To sum up:
- **Xception** does not explicitly use channel or spatial attention mechanisms.
- **CBAM** enhances feature representations by applying **both channel attention** and **spatial attention**, thus providing a more refined approach to attention compared to using only channel attention.
```

> Channel Attention Module : 먼저 입력 피처 맵에 대해 채널별로 중요한 정보를 학습하고, 채널별로 가중치를 적용
> - https://zzziito.tistory.com/52

> Spatial Attention Module : 채널 어텐션이 적용된 피처 맵에 대해 공간적으로 중요한 영역을 학습하여 위치별로 다른 가중치를 적용
> - https://zzziito.tistory.com/53

> Global Average Pooling(정보를 압축) + Global Max Pooling(가장 의미 있는 정보 추출)

> 결국 점점 무거워지는 딥러닝 모델을 경량화시키려는 것이 attention이다.
