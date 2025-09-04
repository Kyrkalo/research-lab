from dataloaders.mnistDataLoader import MnistDataLoader
from models.mdl_mnist_202520 import Mdl_mnist_202520
from test import MNISTTester
from train import MNISTTrainer
import torch.optim as optim
import matplotlib.pyplot as plt
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = Mdl_mnist_202520()
model.to(device)

mnist_config = {
    "learning_rate": 0.01,
    "momentum": 0.5,
    "n_epochs": 100,
    "batch_size_train": 128,
    "batch_size_test": 1000,
    "log_interval": 10,
    "random_seed": 1,
    "device": device,
    "data_dir": "./data"
}

print("Using device:", device)

torch.backends.cudnn.enabled = False
torch.manual_seed(mnist_config["random_seed"])

print("setup MNIST data loader")
train_loader, test_loader = MnistDataLoader(mnist_config).Get()
optimizer = optim.SGD(model.parameters(), lr=mnist_config["learning_rate"], momentum=mnist_config["momentum"])

train = MNISTTrainer(model, optimizer, train_loader, mnist_config)
test = MNISTTester(model, test_loader, device)

print("Start training")
for epoch in range(1, mnist_config["n_epochs"] + 1):
    train.train(epoch)
    test.test(epoch)

# train_losses, train_counter = train.get_train_stats()
# test_losses, test_counter = test.get_test_stats()

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig