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

# Start of Training Code              
net = Net()
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

BATCH_SIZE = 100

EPOCHS = 1

for epoch in  range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        #print(i, i+BATCH_SIZE)
        
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]
        
        # training_data
        net.zero_grad() # zero gradient on each training_data
        outputs = net(batch_X)
        
        loss = loss_function(outputs, batch_y)
        loss.backward() 
        optimizer.step()
       
print(loss)

# Predict on the model
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        if predicted_class == real_class:
            correct += 1
        total += 1
        
print("Accuracy:", round(correct/total, 3)) 