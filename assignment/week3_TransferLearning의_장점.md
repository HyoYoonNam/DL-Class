# Transfer Learning; 전이 학습

Transfer Learning이란 pre-trained neural network를 기반으로 하여 내가 원하는 dataset으로 `재학습` 시키는 것이다.

2015년에 발표된 이미지 분류용 ResNet-152는 약 6,020만 개의 trainable parameter가 존재하고, 8개의 NVIDIA Tesla K80 GPU(1개당 $5,000)를 사용하여 약 2~3주간 학습했다.

학습해야 할 dataset의 크기는 점점 커지고, Neural Network는 점점 더 깊어지는 상황에서(= trainable parameter가 많아지는 상황) parameter들을 random initialized value로부터 시작해서 back propagation을 진행하는 것은 너무나도 많은 금전적, 시간적 비용을 필요로 한다.

다음과 같은 예를 살펴보자.

- 훈련에 2주가 걸리는 dataset D1이 있다고 하자. D1 dataset은 인터넷에 존재하는 수 백만 개의 의류 sample들로 이루어져 있다.
- dataset D2는 어떤 회사 A에서 판매하는 의류 sample들 수 천개로 이루어져 있다(수 천개는 Neural Network로 제대로 된 학습을 하기에는 적은 양이다).
    
    (이 예시에서는 D1과 D2가 정확한 생김새나 종류는 다르겠지만, ‘의류’라는 공통점을 가진다. 이런 경우를 “D1과 D2의 domain이 유사하다”고 한다.)
    
- 충분한 data가 있는 D1을 학습하는 경우에는 layer를 점점 깊게 구성하면서 sample들의 일반적인 특징부터 추상적인 특징까지 골고루 학습할 수 있다.
- 하지만 data가 충분하지 않은 D2를 학습하는 경우에는
    - epoch가 적은 경우:
        
        의류 sample의 종류를 구분하기 위한 특징들을 미처 다 학습하지 못해서 underfitting될 확률이 높다.
        
    - epoch가 많은 경우:
        
        적은 양의 sample들에 대해서 반복적으로 학습함으로 인해 overfitting될 확률이 높다.
        
- 우리는 이를 **transfer learning**으로 해결할 수 있다.
    
    우리는 앞서 D1과 D2의 domain이 유사하다라는 사실을 알아냈다.
    
- D1을 학습한 모델에서는 ‘의류’라는 것에 대해서 충분히 학습했다.
    
    결국 상의는 어떤 상의인지에 관계없이 ‘목부분’이 존재하고, ‘어깨선’도 존재하고, ‘양 팔’도 존재하고, ‘몸통 부분’도 존재하고, …
    
    그리고 하의는 ‘허리 부분’이 존재하고, ‘양 다리’가 존재하고, …
    
- D2의 ‘의류’들도 결국 D1에서 학습한 특징들을 가지고 있다.
- D1의 model에 총 10개의 layer가 있다고 할 때
    - layer가 얕은 부분에서는 ‘목부분’, ‘어깨선’, … 등의 `일반적인 특징`을 학습하고,
    - layer가 깊어질수록 ‘목부분의 레이스’, ‘목부분의 카라’, ‘어깨선의 절개’ 등의 `추상적인(구체적인) 특징`을 학습한다.
- D2의 ‘의류’들은 D1 model이 학습한 `일반적인 특징`에는 대부분 부합할 것이고, `추상적인 특징`에 대해서는 별로 부합하지 않을 것이다.
- 따라서 D2의 model을 학습할 때도 어차피 `일반적인 특징`에 대해서 학습을 할 건데 이미 이 `일반적인 특징`을 너무나도 잘 학습한 D1 모델이 존재한다.
    
    앞에서 `일반적인 특징`은 얕은 layer에서 주로 학습한다고 했던 것을 기억하자.
    
- 그러면 transfer learning을 사용하여 이미 학습된 D1 모델을 그대로 가져온다.
    - `일반적인 특징`에 대해서 더 이상 학습할 필요 없음. 따라서 해당 학습에 걸리는 시간 0초
        
         (사실 나중에 조금은 학습해야 성능이 더 좋아지는데 일단은 쉬운 이해를 위해서 극단적으로 생각하자.)
        
    - 그리고 D1의 깊은 layer는 `추상적인 특징`을 학습했는데 이건 어차피 D2 sample들에게는 거의 부합하지 않으므로 없애버린다(마지막 3개 layer를 없앴다고 가정).
    - 그리고 D2 sample의 세부 분류를 결정적으로 구분하기 위한 layer들을 추가한다(3개 layer를 추가했다고 가정).
        
        결정적으로 구분하기 위해서는 `추상적인 특징`을 학습해야 하기 때문에 가장 마지막 단계의 layer로 추가 하는것이 적합하다.
        
    - 이제 학습을 할 단계이다.
        - 기존에는 10개의 layer를 모두 학습해야 했지만, transfer learning을 통해서 얕은 layer 7개에 대한 좋은 학습 결과물(가중치 parameter)을 이미 가지고 있다.
        - 따라서 3개의 깊은 layer에 대해서만 학습하면 되기 때문에 더욱 집중적으로 학습할 수가 있다.

# 1. 들어가며

## 1.1. 전통적인 machine learning type

전통적인 Machine Learning 분야는 다음과 같이 세 가지로 구분할 수 있다.

- **Supervised Learning(지도 학습)**
    
    Supervised learning is a type of machine learning where an algorithm is trained on labeled data. 
    This means each training example includes an input and a desired output. 
    The algorithm learns to map inputs to outputs by finding patterns in the data. 
    It's commonly used for tasks like `classification` and `regression`.
    
- **Unsupervised Learning(비지도 학습)**
    
    Unsupervised learning involves training an algorithm on data that doesn't have labeled responses. 
    The goal is for the model to identify underlying structures or patterns within the data on its own. 
    It's often used for `clustering`, `dimensionality reduction`, and `association rule learning`.
    
- **Reinforced Learning(강화 학습)**
    
    Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving feedback in the form of rewards or penalties. 
    The agent aims to learn a strategy that maximizes cumulative rewards over time.
    This approach is commonly used in `robotics`, `game playing`, and `navigation tasks`.
    

## 1.2. data labeling의 어려움

**Supervised Learning**은 ****답을 알고 있는 상태에서 학습하기 때문에 비교적 학습이 쉽다.

하지만 labeling하는 작업은 기본적으로 시간과 비용이 많이 든다. quality가 높은 labeling을 위해서는 수 배는 더 많은 시간과 비용이 든다.

## 1.3. self-supervised learning의 등장

**self-supervised learning**은 사실 unsupervised learning를 응용한 것이다.

인터넷에는 `labeled data`는 적지만 `un-labeled data`는 너무나도 충분하게 많다.

<aside>
💡

Transfer learning을 NLP에 적용한 model은 google BERT이다.

BERT는

- 본인이 실제 예측하고 싶은 분야에 대한 train dataset이 있을 때 model에서 이를 바로 학습하는 것이 아니고, wikipedia 같은 곳에서 그냥 일상적인 NL 문장들에 대해서 학습한다.
    - 이때 masked learning을 사용한다(→ labeling resource 없어도 되는 이점)
    - 학습을 완료하면 output layer는 제거하고 hidden layers만 남긴다.
- 위 결과물에 추가로 hidden layer 및 본인이 실제 예측하고 싶은 분야에 대한 output layer를 추가한다.
    
    이때 특정 분야(작업)에 대해 model을 미세 조정하는 것을 fine tuning이라고 한다.
    
    - 그리고 가지고 있던 train dataset을 학습시킨다(최소 1k-2k의 labeled data 필요).
    - 이때는 기존에 wikipedia 학습에서 배웠던 layer들을 이용하여 train dataset과 어떤 차이가 있는지를 중점적으로 학습한다.
- masked learning을 통해 별도의 labeling 없이 학습할 수 있으므로 인터넷에 있는 대량의 NLP 자료로 전체 구조를 학습할 수 있다.
    
    따라서 실제 작업에 대한 훈련을 할 때 비교적 적은 train dataset을 가지고 있다고 하더라도 안정적으로 학습할 수 있다.
    
</aside>

pre-trained dataset과 train dataset의 domain이 유사할수록 fine tuning이 덜 필요하다.