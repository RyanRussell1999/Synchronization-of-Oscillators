import torch

x = torch.tensor([5,3])
y = torch.tensor([2,1])

print(x*y)

x = torch.zeros([2,5])

print(x) 

print(x.shape)

y = torch.rand([2,5])

print(y)

# Reshape
y.view([1,10])

print(y)

# Need to assign to save
y = y.view([1,10])

print(y)