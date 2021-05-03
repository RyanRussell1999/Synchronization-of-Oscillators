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

# Part 3 Start

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init()
        # self.fc1 = nn.Linear(input, output)
        self.fc1 = nn.Linear(28*28, 64) # Input is data input and output is number of neurons in next layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # Number of classes out
    
    # For Feed Forward NN
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Limits data from 0 to 1
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        # Don't want to run RELU on last one as it wants output in range of desired output
        # Use Softmax for probabilities of classification in this case
        return x

net = Net()
X = torch.rand((28,28))
X = X.view(1,28*28)
output = net(X)





