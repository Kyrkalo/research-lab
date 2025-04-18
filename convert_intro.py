import os
import cv2
import numpy as np
from tqdm import tqdm


REBUILD_DATA = True

class DogsVsCats():
    IMG_SIZE = 50  # 50x50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = { CATS: 0, DOGS: 1 }

    training_data = []
    catcount = 0
    dogcount = 0

    def __init__(self):
        pass

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass
        try:
            np.random.shuffle(self.training_data)
            # np.save("training_data.npy", self.training_data, allow_pickle=True)
        except Exception as e:
            print("Error saving training data: ", e)
        finally:
            print("Cats: ", self.catcount)
            print("Dogs: ", self.dogcount)

if REBUILD_DATA:
    dogs_vs_cats = DogsVsCats()
    dogs_vs_cats.make_training_data()

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        n_size = self._to_linear[0] * self._to_linear[1] * self._to_linear[2]

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
        pass

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] 
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

    def load_data(self):
        try:
            self.training_data = np.load("training_data.npy", allow_pickle=True)
            print("Data loaded successfully.")
        except Exception as e:
            print("Error loading data: ", e)
