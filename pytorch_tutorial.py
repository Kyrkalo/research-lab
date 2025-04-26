import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        super().__init__()
        xy = np.loadtxt('data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) #features are in the second column onwards
        self.y = torch.from_numpy(xy[:, [0]]) #labels are in the first column
        self.n_samples = xy.shape[0] #number of samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
    
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4) #number of iterations per epoch
print(f'number of iterations per epoch: {n_iterations}')

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass
        
        # compute loss
        # backward pass
        # update weights
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, inputs: {inputs}, labels: {labels}')