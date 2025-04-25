import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

print(f'0) n_samples: {n_samples}, n_features: {n_features}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
print("scaled data")
print(f'1) X_train shape: {X_train.shape}')
print(f'2) X_test shape: {X_test.shape}')
print(f'3) y_train shape: {y_train.shape}')
print(f'4) y_test shape: {y_test.shape}')

# convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
print("converted to tensors")
print(f'5) X_train shape: {X_train.shape}')
print(f'6) y_train shape: {y_train.shape}')
print(f'7) X_test shape: {X_test.shape}')
print(f'8) y_test shape: {y_test.shape}')

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
print("reshaped y_train and y_test")

#model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)
#loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 1000

for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

with torch.no_grad():
    y_test_predicted = model(X_test)
    y_test_predicted_cls = y_test_predicted.round()
    acc = y_test_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc.item():.4f}')

