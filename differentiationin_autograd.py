import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

Q = 3*a**3 - b**2
print(Q)
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
# check if collected gradients are correct
print(f'{9*a**2 == a.grad};  a.grad: {a.grad}')
print(f'{-2*b == b.grad};  b.grad: {b.grad}')