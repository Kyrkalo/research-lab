import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare the data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape y to be a column vector
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

# model
input_size = n_features
output_size = 1
model = nn.Linear(output_size, 1)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    #update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

#plot
# detach() is for removing the tensor from the computational graph
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro', label='Original data')
plt.plot(X_numpy, predicted, label='Fitted line')
plt.plot(X_numpy, predicted, 'b-')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
# Save the model