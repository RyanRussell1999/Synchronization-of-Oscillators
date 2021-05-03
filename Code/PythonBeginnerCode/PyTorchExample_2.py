import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Most of your time will be spent gathering data, organizing data, 
# and putting it in the correct format for training.

# Data doesn't really apply to DQN because we are using this not to
# identify data but to take states and determine the best control input.


# MNIST is 28x28 hand drawn images

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# Batch Size (How much data we want to pass to the NN at one time)
# Deep Learning has advantage when dealing with large amounts of data - need to batch to deal with large amounts of data
# Batching also limits overfitting
# Want to have GENERALIZATION - so that it can work best with new data

for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]

print(y)

plt.imshow(data[0][0].view(28,28)) # shape is 1x28x28

# Want equally varied data so that it doesn't assume the most common input 

total = 0 
counter_dict = (0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0) 

for data in trainset:
    Xs, ys = data
    for y in ys 
        counter_dict[int(y)] += 1
        total += 1

for i in counter_dict
    print(f"{i} : {counter_dict[i]/total*100}")



