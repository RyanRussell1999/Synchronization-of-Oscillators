import torch

x = torch.Tensor([5,3])
y = torch.Tensor([2,1])

print(x*y) # Shows the multiplication of tensors (arrays) 

x = torch.zeros([2,5])

print(x) 

x.shape

y = torch.rand([2,5])

print(y)

y = y.view([1,10]) # Reshape the array

print(y)

# Doing math with arrays on GPU
# Library to help do array math