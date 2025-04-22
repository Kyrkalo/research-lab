import numpy as np

x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

w  = 0.0

#model prediction
def forward(x):
    return w * x

#loss
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

#gradient
#MSE = 1/n * (y_pred - y) ** 2
#derivative of MSE = 1/n 2x (y_pred - y)

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    #forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    #gradient
    dw = gradient(x, y, y_pred)

    #update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'Epoch {epoch+1}: w = {w}, loss = {l}')


print(f'Prediction before training: {forward(5)}')