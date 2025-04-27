import torch
import torch.nn as nn

class NeuralNet1(nn.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        return
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return torch.sigmoid(out)
    
class NeuralNet2(nn.Module):
    def __init(self,input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 =nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        return

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no softmax functiion here, because nn.CrossEntropyLoss() applies softmax internally
        return out
    
model1 = NeuralNet1(28*28, 5, 1)
criterion = nn.BCELoss()

model2 = NeuralNet2(28*28, 5, 3)
criteriopn = nn.CrossEntropyLoss()