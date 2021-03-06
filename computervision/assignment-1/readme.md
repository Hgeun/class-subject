교재에 있는 Algorithm 2.3 (Generation of additive, zero mean Gaussian noise)을 구현하시오.  
이를 이용하여 합성하여 얻은 잡음 영상에 대해, 아래 과정을 보이시오.  
8 비트로 나타내는 흑백 디지털 영상에 평균이 0 이고 표준 편차가 σ인 가우시안  
잡음을 더하여 얻은 합성된 서로 다른 두 장 원 영상의 합성 잡음 영상을 고려하자   
(이 과정에서 0 과 255 값에서 clipping 이 필요하다).  
세 가지 서로 다른 표준 편차 값에 대해 표준 편차 추정과정의 결과를 보이고,  
원 영상 자체의 신호 성분의 표준 편차 값과의 크기 관계를 고려하여 설명하시오.  
  
a) (20) 원 영상을 안다고 할 때 합성 잡음 영상으로부터 표준 편차 σ를 추정하여, 잡음 영상을  
합성 시 사용한 원래의 표준 편차 값과 비교하여 clipping 의 영향에 대해 토의하시오.  
  
b) (20) 원 영상을 모른다고 할 때, 주어진 합성 잡음 영상에서 상대적으로 평탄한 영역을  
사용자가 보고 설정하여 잡음의 표준편차 σ를 추정하시오 (평탄한 영역을 어디에 전체 화소  
수의 몇 %되게 설정했는 지도 아울러 보이시오). 추정한 잡음의 표준편차를 살펴볼 때, 이 추정  
과정이 만족스러운 입력 영상과 그렇지 않은 영상의 (앞에서 두 가지 실험영상과 합성 영상을 만들  
때 사용하는 세 가지 표준 편차 값을 선택할 때 이러한 토의과정을 고려하여 선택) 예를 들어  
입력영상의 특성을 고려하여 비교, 설명하시오.  
