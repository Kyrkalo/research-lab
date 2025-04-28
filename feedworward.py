import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 2

input_size = 28*28 #784
hidden_size = 100
num_classes = 10

batch_size = 100
learning_rate = 0.001

#MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

#torch.Size([100, 1, 28, 28]) torch.Size([100])
# 100 samples, 1 channel, 28 height, 28 width

# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #input layer to hidden layer
        self.relu = nn.ReLU() #activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) #hidden layer to output layer

    def forward(self, x):
        #out = x.view(-1, input_size) #flatten the input
        out = self.fc1(x) #input layer to hidden layer
        out = self.relu(out) #activation function
        out = self.fc2(out) #hidden layer to output layer
        # not using softmax here, as it is included in the loss function (CrossEntropyLoss)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optimizer

#training the model
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        #loss calculation
        loss = criterion(outputs, labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print statistics
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #get the index of the max log-probability
        n_samples += labels.size(0) #number of samples
        n_correct += (predicted == labels).sum().item() #number of correct predictions
        
    print(f'Accuracy of the model on the 10000 test images: {100.0 * n_correct / n_samples} %')