pytorch tutorial  
[Warm-up: numpy](https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_numpy.html?highlight=numpy)  


```python
import numpy as np
import matplotlib.pyplot as plt

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
lossep = []

# Create random input and output data
x = np.random.randn(N, D_in) #랜덤으로 입력 데이터 생성, randn => normal distribution으로 데이터 생성
y = np.random.randn(N, D_out) #랜덤으로 비교할 타겟 데이터 생성

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # h = relu(w1*x)
    # y_pred = w2*h_relu

    # Compute and print loss
    # loss = sigma(i=1~n)[ (y_pred - y)^2 ]
    loss = np.square(y_pred - y).sum()
    lossep.append(loss)
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # loss를 미분 진행
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    
plt.plot(lossep)
```
