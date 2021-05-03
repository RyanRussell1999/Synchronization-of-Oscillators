# Network Analysis and Visualizations - 8

# ON GPU - Convert NN to being on GPU - 7

# Convolutional NN 
# Used for Image tasks
#          Sequential Data

# Can have 2D or 3D inputs -- We only need 1D for cart pole

# Convolution - uses windows to determine "features" of the numbers
#           Slides and Condenses Image each time to determine features #           at each layer

# Pooling - Takes values from a window and pools them together

# Classify Cat vs Dog 


# Preprocessing Data
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


MODEL_NAME = f"model-{int(time.time())}"
REBUILD_DATA = False # Flag so you don't rebuild data a ton

class DogsVSCats():
    IMG_SIZE = 50 # Make Images all 50x50 - Need uniform input
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1} # Give each data set a class 
    # This gives a 1 hot vector - we may not use this
    
    training_data = []
    catcount = 0 # PAY Attention to Balance
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                # Try to use data or if doesn't work then pass it
                try: 
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
                    # Doesn't use color because not a relevant feature
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    
                    # Keep track of count of data
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1

                except Exception as e:
                    pass
            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
            print("Cats:", self.catcount)
            print("Dogs:", self.dogcount)
            
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)


# Define GPU Devices
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(input, output, window size (5x5))
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        #print(x[0].shape)
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]  
        return x
        
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Function to Test (in and out of sample) and Train data
def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss
    
# Test Function    
def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,1,50,50).to(device),y.to(device))
    return val_acc, val_loss
    
# Train Function
def train():
    print("Model Name:", MODEL_NAME)
    BATCH_SIZE = 100
    EPOCHS = 8
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            print("EPOCH:",epoch)
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)
                
                acc, loss = fwd_pass(batch_X, batch_y, train=True) # Train
                
                if i % 50 == 0:
                    val_acc, val_loss = test(size = 100) # Validate
                    f.write(f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
                    
def create_acc_loss_graph(MODEL_NAME):
    contents = open("model.log", "r").read().split('\n')
    
    times = []
    accuracies = []
    losses = []
    
    val_accs = []
    val_losses = []
    
    for c in contents:
        if MODEL_NAME in c:
            name, timestamp, acc , loss, val_acc, val_loss = c.split(",")
            
            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))
            
            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))
            
    fig = plt.figure()
    
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
    
    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_accs")
    ax1.legend(loc=2)
    
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_losses")
    ax2.legend(loc=2)
    
    plt.show()
        

# Start of Training Code              
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001) # all parameters are contolled by adam optimizer
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

# Decide the data to train and test
VAL_PCT = 0.1 # test 10% of data
val_size = int(len(X)*VAL_PCT) # Validation test size

# Separate Testing and Training Data
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# Train NN
train()
create_acc_loss_graph(MODEL_NAME)

# When losses or accuracies diverge then you know you are over training the NN


