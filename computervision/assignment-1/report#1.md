원본 흑백 영상 | 노이즈 합성 영상
-- | --
평균 : 105.87   표준 편차 : 47.67 | 표준편차   30 | 표준편차   100 | 표준편차   150
  
Top ( left : original   image, right : noise std=30)   Bottom ( left : noise   std=100, right : noise std=150)

![image](https://user-images.githubusercontent.com/33209778/45472290-c12c6d80-b76e-11e8-9e0f-2309f36cfbcb.png)  ![image](https://user-images.githubusercontent.com/33209778/45472301-c689b800-b76e-11e8-9585-ea75d7d97f8a.png)  
![image](https://user-images.githubusercontent.com/33209778/45472307-ca1d3f00-b76e-11e8-803d-e2a1d86e1d52.png)  ![image](https://user-images.githubusercontent.com/33209778/45472311-cc7f9900-b76e-11e8-9c53-7f081977b0bb.png)  
  
a) 원 영상을 알고 있을 때, 원 영상과 노이즈를 합성한 영상, 두 영상사이의 차 영상을 통해 노이즈의 표준 편차를 추정해볼 수 있다. 추정 과정을 살펴보면 다음과 같다.
	1) 차 영상을 구한다. (시각화를 위해 차 영상을 [0,255] 범위로 normalize.)
  
Top  
- std 30 노이즈합성영상 – 원영상  
  
Bottom  
- Left : std 100 노이즈합성영상 – 원영상  
- Right : std 150 노이즈합성영상 – 원영상  

![image](https://user-images.githubusercontent.com/33209778/45472486-53347600-b76f-11e8-9d0a-63db63c41c30.png)  
![image](https://user-images.githubusercontent.com/33209778/45472494-5891c080-b76f-11e8-9fda-a66cb36b50ad.png) ![image](https://user-images.githubusercontent.com/33209778/45472502-5c254780-b76f-11e8-9b28-d8b9366da8a9.png)
  
Std30노이즈합성영상–원영상 | Std100노이즈합성영상–원영상 | Std150노이즈합성영상–원영상
-- | -- | --
평균=0.0837, std=29.8592 | 평균=5.3539, std=78.7586 | 평균=8.8723, std=95.9086
  
위의 결과를 살펴보면 std30일 때 29.8592로 거의 비슷한 표준편차를 추정하며 Std100, std150에서는 큰 오차를 보이는 것을 알 수 있다. 이를 clipping의 영향의 관점으로 분석해보자. 
std30에서는 차영상의 평균이 0에 근접하지만, std100과 std150에서는 평균이 커짐을 알 수 있다. 이는 노이즈합성영상을 만드는 과정에서 noise를 원영상에 그대로 더할 때 0보다 작은 값을 0으로 clipping하는 경우가 255보다 큰 값을 255로 clipping하는 경우보다 많아서 전체적으로 평균이 오르는 결과를 보이게 된다. 또한 원영상의 평균이 105.87이므로 표준편차가 30인 가우시안 노이즈는 [0,255]범위 안에서 충분히 수용 가능한 반면, 표준편차가 100, 150인 가우시안 노이즈는 0~255범위에서 벗어나는 경우가 점점 많아지게 된다.
  
  
b) 원 영상을 모르는 경우 상대적으로 평탄한 지역의 평균과 표준편차를 이용하여 노이즈를 추정할 수 있다. 위 영상에서 평탄한 지역으로 왼쪽의 체크무늬 벽면과 중앙의 벽면을 선정하여 노이즈를 추정하였다.  
![image](https://user-images.githubusercontent.com/33209778/45472608-ad353b80-b76f-11e8-8a4f-57bc07e297e8.png)
  
 왼쪽의 체크무늬의 부분을 ‘tile’, 중앙의 벽면을 ‘wall’이라 명명하여 실험을 진행하였다. 각각의 영역의 왼쪽 위 모서리 좌표와 크기는 ‘tile’이 (43,13) (38,40) 이고, ‘wall’이 (357,2) (155,161) 이다. 전체에 대한 비율은 ‘tile’이 0.0039, ‘wall’이 0.0647 이다. ‘wall’이 ‘tile’에 비해 약 16.6배 큰 비율을 갖는다.
 각 노이즈 합성 영상에서의 ‘tile’ 부분을 살펴보면 다음과 같다. 왼쪽부터 차례대로 노이즈의 표준편차가 std30, std100, std150인 경우이다.

평균=173.6304/std=30.8349 | 평균=162.3458/std=78.1813 | 평균=156.3533/std=93.5279
-- | -- | --
![image](https://user-images.githubusercontent.com/33209778/45472625-be7e4800-b76f-11e8-9f9c-547ac05a660c.png) | ![image](https://user-images.githubusercontent.com/33209778/45472627-c0e0a200-b76f-11e8-897a-937119c70620.png) | ![image](https://user-images.githubusercontent.com/33209778/45472630-c4742900-b76f-11e8-8499-5a5872ae44f0.png)

전체에서 약 0.39%의 크기 비율을 갖는 ‘tile’부분에서 구한 local 평균과 표준편차는 위와 같았다. 왼쪽에서 오른쪽으로 갈수록 0과 255로 saturate된 픽셀이 많아지는 것을 확인할 수 있다. Std30에서 30.8349로 거의 같은 표준편차를 추정할 수 있었고, std100에서 약 22, std150에서 약56.5의 오차로, 점점 원래 노이즈의 표준편차와 차이를 보이게 된다. 
 
마찬가지로 ‘wall’부분을 살펴보면 다음과 같다. 왼쪽부터 차례대로 노이즈의 표준편차가 std30, std100, std150인 경우이다.  

평균=151.5610/std=30.0551 | 평균=146.3811/std=81.4443 | 평균=141.4219/std=96.0558
-- | -- | --
![image](https://user-images.githubusercontent.com/33209778/45472748-14eb8680-b770-11e8-90d7-5f0b2f02c353.png) | ![image](https://user-images.githubusercontent.com/33209778/45472755-19b03a80-b770-11e8-9867-8aebf76c6e65.png) | ![image](https://user-images.githubusercontent.com/33209778/45472762-1f0d8500-b770-11e8-91de-8f16e331be28.png)  

‘tile’에서의 평균보다 전체적으로 커진 것을 확인할 수 있다. 표준편차의 오차도 더 줄어들었다. 로컬이지만 전체에서 차지하는 비율이 약 6.47%로 ‘tile’보다 크므로, 더 많은 원본의 경향성을 가져갈 수 있는 것으로 분석할 수 있다.
Std30과 다르게 std100과 std150에서의 추정 표준편차는 오차가 매우 크다. 단, 추정 표준편차가 실제 표준편차에 조금 더 가까워진 것을 확인할 수 있는데 이는 앞서 a)에서 설명한 clipping에 이어서, ‘wall’ 로컬영역의 intensity의 평균이 전체 원본 이미지의 평균 보다 약간 더 큰 크기의 노이즈를 수용할 수 있기 때문이라고 볼 수 있다.







