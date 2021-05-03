import torch
import torchvision # Collection of vision data for pytorch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Most of time spent on data itself -- In our case need to collect the information 
# from the cart-pole

# Two main datasets - Training and Testing to see if not overfit (AKA works in practice)
train = datasets.MNIST("", train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# Creating NN Section

class Net(nn.Module):
    def __init__(self):
        super().__init__() # Init nn module
        
        # Defining Layers
        # self.fc1 = nn.Linear(input, output)
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        
    # Defines how layers will be passed    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1) # Get probability distribution on output
        return x
               
net = Net()

X = torch.rand((28,28))
X = X.view(1,28*28) # Resize for NN

output = net(X)

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and labels
        X, y = data # X is data and y is class
        





