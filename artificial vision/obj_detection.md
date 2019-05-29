# Obj detection
Faster RCNN, YOLO, SSD

## Size of Receptive Field
- 네트워크를 역으로 거슬러 올라가면 알 수 있음
- 3x3 conv를 거슬러 올라가면 receptive field size가 2씩 증가 (3x3 Fmap) -> [3x3 Conv] -> (1x1 Fmap)  

CNN -> Fmap ->

## Bounding box
=Anchor box
미리 모양에 대한 prior를 주고 훈련

## feature pyramid network
![image](https://user-images.githubusercontent.com/33209778/58526521-2abfaf80-820a-11e9-9bd5-fddd4d593aa3.png)  
