import torch
import torchvision
from torch.utils.data import DataLoader
import os
from pathlib import Path

class MnistDataLoader:

    def __init__(self, config):
        self.batch_size_train = config["batch_size_train"]
        self.batch_size_test = config["batch_size_test"]
                        
        BASE_DIR = Path(__file__).resolve().parent
        RESULTS_DIR = BASE_DIR.parent / "results"
        
        self.data_dir = RESULTS_DIR / config["data_dir"]
        pass

    def Get(self):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_loader = DataLoader(
            torchvision.datasets.MNIST(self.data_dir, train=True, download=True, transform=transforms),
            batch_size=self.batch_size_train, shuffle=True
        )
        test_loader = DataLoader(
            torchvision.datasets.MNIST(self.data_dir, train=False, download=True, transform=transforms),
            batch_size=self.batch_size_test, shuffle=False
        )
        return train_loader, test_loader