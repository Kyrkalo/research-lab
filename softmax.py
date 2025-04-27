import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
output = softmax(x)
print(output) # [0.65900114 0.24243297 0.09856589]

x = torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0)
print(output) # tensor([0.6590, 0.2424, 0.0986])