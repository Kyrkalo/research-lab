import torch
import torch.nn as nn

class MulticlassClassifier(nn.Module):  # formerly NeuralNet2
    def __init__(self, input_size, hidden_size, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out  # logits; use CrossEntropyLoss
