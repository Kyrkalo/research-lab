import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork

#convert imnest image files into tensors
transform = transforms.ToTensor()

#train data
train_data= datasets.MNIST(root='./cnn_data', train=True, download=True, transform=transform)

#test data
test_data = datasets.MNIST(root='./cnn_data', train=False, download=True, transform=transform)
# print("Length of train data: ", len(train_data))
# print("Length of test data: ", len(test_data))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

torch.manual_seed(41) # for reproducibility
model = ConvolutionalNeuralNetwork()
# print(model)

#loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0011)
#training the model

import time
start_time = time.time()


#create a variables to track things

num_epochs = 50
train_loss = []
train_correct = []
test_loss = []
test_correct = []

for epoch in range(num_epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred, dim=1) # get the predicted class
        batch_corr = (predicted[1] == y_train).sum() # compare predicted class with actual class
        trn_corr += batch_corr



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if b % 100 == 0:
            print(f" -----> Epoch {epoch+1}/{num_epochs}, Batch {b}, Loss: {loss.item()}")
           
        train_loss.append(loss.item())
        train_correct.append(trn_corr.item())




current_time = time.time()
total_time = current_time - start_time
print("Total time taken for training: ", total_time / 60, " minutes")


# with torch.no_grad(): # No need to track gradients for testing
#     print("Testing the model")
#     for b, (X_test, y_test) in enumerate(test_loader):
#         y_test_pred = model(X_test)

#         loss = criterion(y_test_pred, y_test)        
#         predicted = torch.max(y_test_pred, dim=1)

#         test_loss.append(loss.item())
#         test_correct.append(predicted[1].eq(y_test).sum().item())

plt.plot(train_loss, label='Train Loss', color='blue')
# plt.plot(test_loss, label='Test Loss', color='green')
# plt.plot(train_correct, label='Train Correct', color='red')
# plt.plot(test_correct, label='Test Correct', color='black')
# plt.title('Accuracy')
plt.legend()
plt.show()

#grab the image
# test_data[4143] # tensor of shape (1, 28, 28) and label 4

#grab just the data
#reshape to (28, 28)
img = test_data[4143][0] # tensor of shape (1, 28, 28)
img = img.view(28, 28) # tensor of shape (28, 28)

model.eval() # set the model to evaluation mode
with torch.no_grad():
    new_pred = model(test_data[4143][0].view(1,1, 28, 28)) # add batch dimension

new_pred = torch.max(new_pred, dim=1)[1] # get the predicted class
print("Predicted class: ", new_pred.item())
print("Actual class: ", test_data[4143][1])

torch.save(model.state_dict(), 'model.pth') # save the model
