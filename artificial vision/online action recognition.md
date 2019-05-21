# Online Action Recognition using CNN-LSTM and VAE

## 1. Dataset
### 1-1) THUMOS’14,15
![image](https://user-images.githubusercontent.com/33209778/58100170-154df280-7c18-11e9-8fc4-aeaf147640fe.png)
- UCF101 dataset을 포함  
- 실시간 행동 인식을 위해서 20개의 클래스에 대해 temporal annotation 제공  
- Dataset download : [THUMOS 2014](https://www.crcv.ucf.edu/THUMOS14/download.html), [THUMOS 2015](http://www.thumos.info/download.html)

### 1-2) UCF-Crimes
![image2](https://webpages.uncc.edu/cchen62/img/thumbnail/ucf-crime.png)
- UCF(University of Central Florida)에서 발표
- 실제 13종류의 범죄 영상과 평범한 영상을 포함
- 각 클래스는 Abuse, Arrest, Arson, Assault, Road accident, Burglary, Explosion, Fighting, Normal, Robbery,  
Shooting, Stealing, Shoplifting, Vandalism 총 14개
- 각 영상은 다양한 길이로 구성
- Dataset download : [UCF-Crimes](https://webpages.uncc.edu/cchen62/dataset.html)

## 2. Network
### 2-1) video action recognition을 위한 여러 네트워크 구조들
![image](https://user-images.githubusercontent.com/33209778/58101299-5cd57e00-7c1a-11e9-92a8-647f02970de1.png)
![image](https://user-images.githubusercontent.com/33209778/58101305-5fd06e80-7c1a-11e9-9fd0-6bb4c5022d90.png)  
- 이전 프레임들의 정보를 길이의 제한없이 가져갈 수 있는 RNN계열 네트워크 구조(a)가 online action recognition에 적합하다고 판단  

### 2-2) RNN을 이용한 online action recognition 네트워크 구조
#### 2-2-a) Temporal Recurrent Networks for Online Action Detection
- Xu, Mingze, et al. "Temporal Recurrent Networks for Online Action Detection." arXiv preprint arXiv:1811.07391 (2018).
![image](https://user-images.githubusercontent.com/33209778/58101877-5398e100-7c1b-11e9-8237-bd802098670b.png)
![image](https://user-images.githubusercontent.com/33209778/58101883-55fb3b00-7c1b-11e9-825b-98b41c5f41c3.png)  
- 사람이 현재 행동 인식에 미래 행동 예측을 사용한다는 사실에서 착안하여 미래 행동 예측을 현재 행동 인식에 반영할 수 있는 네트워크 구조 제안
- 미래 예측을 위한 RNN temporal decoder 부분과 현재와 과거 feature들을 종합하여 다시 RNN cell에서 처리

#### 2-2-b) Modeling temporal structure with lstm for online action detection
- De Geest, et al. "Modeling temporal structure with LSTM for online action detection." WACV, 2018.
![image](https://user-images.githubusercontent.com/33209778/58102109-bdb18600-7c1b-11e9-937b-1f3275d90748.png)
- LSTM을 이용하여 실시간 행동 인식을 위한 네트워크 모델 구조를 제안
- 현재 프레임의 CNN feature와 과거 class 확률을 각각 두 LSTM의 입력으로 사용
- Breakfast[9], TV series[10] dataset 사용하여 실험.
- 일반적으로 쓰이는 dataset에 대한 실험 부족

#### 2-2-c) RED:Reinforced Encoder-Decoder Networks for Action Anticipation
- Gao, Jiyang, Zhenheng Yang, and Ram Nevatia. "Red: Reinforced encoder-decoder networks for action anticipation." arXiv preprint arXiv:1707.04818 (2017).  
![image](https://user-images.githubusercontent.com/33209778/58102214-e20d6280-7c1b-11e9-8c2e-8f864178a2bd.png)
- LSTM을 이용하여 다음 행동을 예측하기 위한 encoder-decoder를 구현
- 강화학습기를 이용하여 classifier 학습
- TV series[10], THUMOS’14[2] dataset 사용
- 기존의 행동예측과 다르게 고정된 시간의 미래가 아닌 유동적으로 시간을 조정하여 행동 예측 가능

### 2-3) 실험하고자 하는 네트워크 구조
- 미래 feature를 얻고자 VAE 학습  
![image](https://user-images.githubusercontent.com/33209778/58102440-3e708200-7c1c-11e9-8ab9-9b2fc6367ee1.png)
- CNN-LSTM 구조와 VAE를 결합  
![image](https://user-images.githubusercontent.com/33209778/58102633-8d1e1c00-7c1c-11e9-8fbb-70287cbd6254.png)
