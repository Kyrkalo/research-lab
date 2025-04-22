import torch

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

learning_rate = 0.1
n_iters = 1000

for epoch in range(n_iters):
    # Forward pass
    y_pred = forward(x)

    # Compute loss
    l = loss(y, y_pred)

    # Backward pass (compute gradients)
    l.backward()

    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

        # Zero the gradients after updating weights
        w.grad.zero_()

    if epoch % 2 == 0:
        print(f'Epoch {epoch+1}: w = {w.item()}, loss = {l.item()}')

print(f'Prediction after training: {forward(5)}')