import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([4, 5, 6])

x, y = torch.meshgrid(a, b)
print(x)
print(y)

x = x.reshape(-1)
y = y.reshape(-1)
print(x)
print(y)
z = torch.stack([x, y, x, y], dim=1)
print(z)
print(z.view(-1,1,4))