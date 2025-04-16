import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

train = datasets.MNIST("data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("data", download=False, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim = 1)
        return x
    
net = Net()
#print(net)

X = torch.rand((28*28))
X =X.view(-1, 28*28)
##print(X)

output = net(X)
#print(output)

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for ixd, i in enumerate(output):
            correct += 1 if torch.argmax(i) == y[ixd] else 0                                     
        total += 1
print(f"Accuracy: {round(correct/total, 3)}%")

print(torch.argmax(net(X[1].view(-1, 28*28))[0]))

import matplotlib.pyplot as plt
plt.imshow(X[1].view(28, 28))
plt.show()
torch.save(net.state_dict(), "model.pth")
#plt.imshow(data[0][0][0].numpy(), cmap='gray')
# plt.imshow(data[0][0].view(28, 28))
# plt.show()
# total =0
# counter_dict = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
# for data in trainset:
#     xs, xy = data
#     for y in xy:
#         counter_dict[y.item()] += 1
#         total += 1

# for i in counter_dict:
#     print(f"{i}: {counter_dict[i]/total*100}")
