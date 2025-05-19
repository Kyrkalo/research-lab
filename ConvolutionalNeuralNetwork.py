import torch
import torch.nn as nn
import torch.nn.functional as F



# model class
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # convolutional layers
        # 1st layer: input channels = 1 (grayscale image), output channels = 6, kernel size = 3x3, stride = 1, padding = 1
        # 2nd layer: input channels = 6, output channels = 16, kernel size = 3x3, stride = 1, padding = 0
        self.convolutional_layer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.convolutional_layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)

        #fully connected layers or pooling layers
        # 1st layer: input features = 16*6*6 (after flattening), output features = 120
        # 2nd layer: input features = 120, output features = 84
        # 3rd layer: input features = 84, output features = 10 (number of classes)
        self.fc1 = nn.Linear(in_features=16*6*6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.convolutional_layer1(x)) # apply ReLU activation function after 1st layer
        x = F.max_pool2d(x, kernel_size=2, stride=2) # apply max pooling with kernel size 2x2 and stride 2

        #second layer
        x = F.relu(self.convolutional_layer2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #Re-view to flatten the output from the convolutional layers
        x = x.view(x.size(0), -1) # flatten the output to (batch_size, num_features)

        x = F.relu(self.fc1(x)) # apply ReLU activation function after 1st fully connected layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # output layer (no activation function applied here, as we will use CrossEntropyLoss which applies softmax internally)
        return x