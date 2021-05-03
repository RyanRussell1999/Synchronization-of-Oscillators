import torch
import torchvision # Collection of vision data for pytorch
from torchvision import transforms, datasets


# Most of time spent on data itself -- In our case need to collect the information 
# from the cart-pole

# Two main datasets - Training and Testing to see if not overfit (AKA works in practice)
train = datasets.MNIST("", train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
 
# Name of the game is generalization!!!!!

# Iterating over data
for data in trainset:
    print(data)
    break

    
x, y = data[0][0], data[1][0]

print(y)

import matplotlib.pyplot as plt

# Plotting a tensor image -- normal size is 1x28x28
plt.imshow(data[0][0].view([28,28]))
plt.show()


# Optimizer only tries to decrease loss
# Doesn't know how to maximize data just takes easiest/quickest route to minimize loss

# Can't train out of holes -- need to have balanced data
# Want dataset to be as balanced as possible -- Amounts of each class are same

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys: 
        counter_dict[int(y)] += 1
        total += 1
        
print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")
    
