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

REBUILD_DATA = True # Flag so you don't rebuild data a ton

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

