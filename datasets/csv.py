import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        super().__init__()
        xy = np.loadtxt(csv_file, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) #features are in the second column onwards
        self.y = torch.from_numpy(xy[:, [0]]) #labels are in the first column
        self.n_samples = xy.shape[0] #number of samples
        self.transform = transform
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor:
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(np.array(inputs)), torch.from_numpy(np.array(target))
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs = inputs * self.factor
        return inputs, target
    
composed = torchvision.transforms.Compose([
    ToTensor(),
    MulTransform(5)
])

dataset = WineDataset(transform=composed)
first_data = dataset[0]
feature, label = first_data
print(feature)
print(type(feature), type(label))