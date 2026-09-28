import torch
import torch.nn as nn
a = torch.tensor([[1, 2], [3, 4]], requires_grad=True , dtype=torch.float32)

y = 3*(a**2) +2
loss = y.sum()
print(y)
print(loss)
loss.backward()
print(a.grad)