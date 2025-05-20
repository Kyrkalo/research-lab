import torch
import torch.nn as nn
import torch.nn.functional as F



# model class
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                 # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)                                        # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)                                                     # 14x14 -> 7x7
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes
    
    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool2(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)             # Flatten
        x = F.relu(self.fc1(x))                # Fully connected
        x = self.fc2(x)                        # Output logits
        return x