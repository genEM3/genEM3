import torch

x = torch.randn(4, 2, 1, 1)
print(x)
x0 = torch.nn.functional.softmax(x, dim=1)
print(x0)
a = 1