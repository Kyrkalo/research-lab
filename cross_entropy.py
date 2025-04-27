import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Good prediction loss: {l1}')
print(f'Bad prediction loss: {l2}')

loss = nn.CrossEntropyLoss()
Y = torch.tensor([2, 0, 1])
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.0, 1.0, 2.1], [2.0, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Good prediction loss: {l1.item()}')
print(f'Bad prediction loss: {l2.item()}')

_, Y_prediction1 = torch.max(Y_pred_good, 1)
_, Y_prediction2 = torch.max(Y_pred_bad, 1)
print(f'Predicted class for good prediction: {Y_prediction1}')
print(f'Predicted class for bad prediction: {Y_prediction2}')