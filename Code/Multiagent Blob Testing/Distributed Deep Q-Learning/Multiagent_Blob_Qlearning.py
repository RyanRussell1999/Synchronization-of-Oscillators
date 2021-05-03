#   Distributed Multiagent Deep Q-Learning - 3 blobs
#   -------------------------------------------
#   Author: Ryan Russell
#   
#   Communication Topology:
#   1 talks to 2
#   2 talks to 3
#   3 talks to 1

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from tqdm import tqdm

import torch
import torchvision # Collection of vision data for pytorch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


style.use("ggplot")

SIZE = 10
EPISODES = 5_000
SHOW_EVERY = 500
STEPS = 50
epsilon = 0.5
EPSILON_DECAY = 0.9998

OBSERVATION_SIZE = 4
OUTPUT_SIZE = 9

LEARNING_RATE = 0.01
DISCOUNT = 0.95

AGENT_N = 1
d = {1: (255, 175, 0)}

# Neural Network 
class Net_1(nn.Module):
    def __init__(self):
        super().__init__() # Init nn module
        
        # Defining Layers
        # self.fc1 = nn.Linear(input, output)
        self.fc1 = nn.Linear(OBSERVATION_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, OUTPUT_SIZE)
        
        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)
        self.loss_f = nn.MSELoss()
        
    # Defines how layers will be passed    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def train(self, output, current_qs):
        self.zero_grad()
        self.loss = self.loss_f(output, current_qs)
        self.loss.backward()
        self.optimizer.step()
        
        
        

class Net_2(nn.Module):
    def __init__(self):
        super().__init__() # Init nn module
        
        # Defining Layers
        # self.fc1 = nn.Linear(input, output)
        self.fc1 = nn.Linear(OBSERVATION_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, OUTPUT_SIZE)
        
        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)
        self.loss_f = nn.MSELoss()
        
    # Defines how layers will be passed    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 
        return x
        
    def train(self, output, current_qs):
        self.zero_grad()
        self.loss = self.loss_f(output, current_qs)
        self.loss.backward()
        self.optimizer.step()

class Net_3(nn.Module):
    def __init__(self):
        super().__init__() # Init nn module
        
        # Defining Layers
        # self.fc1 = nn.Linear(input, output)
        self.fc1 = nn.Linear(OBSERVATION_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, OUTPUT_SIZE)
        
        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)
        self.loss_f = nn.MSELoss()
        
    # Defines how layers will be passed    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 
        return x
        
    def train(self, output, current_qs):
        self.zero_grad()
        self.loss = self.loss_f(output, current_qs)
        self.loss.backward()
        self.optimizer.step()

# Blob Class - Act as Agents for test
class Blob: 
    def __init__(self):
    # Initialize Random Position
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        
    def __str__(self):
        return f"{self.x}, {self.y}"
        
    def action(self, choice):
    # Take an action based on the choice
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=1)
        elif choice == 2:
            self.move(x=1, y=-1)
        elif choice == 3:
            self.move(x=-1, y=-1)   
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1) 
        elif choice == 7:
            self.move(x=0, y=-1)    
        elif choice == 8:
            self.move(x=0, y=0) 
            
    def move(self, x=False, y=False):
    # Move the Blob from the given input or randomly if no input
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
            
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y            
        
    # Stop Blob from going out of boundary
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1  


# Initialize Each Agent's NN

# Define device to run on 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

net_1 = Net_1().to(device)
net_2 = Net_2().to(device)
net_3 = Net_3().to(device)

# Initialize Episode Rewards
episode_rewards_1 = [] 
episode_rewards_2 = [] 
episode_rewards_3 = []
 
# Deep Q-Learning Algorithm 
for episode in range(EPISODES):
    
    print("EPISODE:", episode)

    # Create Agents
    Agent1 = Blob()
    Agent2 = Blob()
    Agent3 = Blob()
    
    # Adjacency Matrix
    A = np.array([[0, 0, 1],[1, 0 ,0], [0, 1, 0]])
    
    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f"on #{episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean 1 {np.mean(episode_rewards_1[-SHOW_EVERY:])}")
        print(f"{SHOW_EVERY} ep mean 2 {np.mean(episode_rewards_2[-SHOW_EVERY:])}")
        print(f"{SHOW_EVERY} ep mean 3 {np.mean(episode_rewards_3[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    episode_reward_1 = 0
    episode_reward_2 = 0
    episode_reward_3 = 0
    
    # Iteratre through model for i steps
    for i in tqdm(range(STEPS)):
    
    # Get Distance apart from each agent for every agent
        obs_1 = torch.FloatTensor([A[0][1]*(Agent1.x - Agent2.x), A[0][1]*(Agent1.y - Agent2.y), A[0][2]*(Agent1.x - Agent3.x), A[0][2]*(Agent1.y - Agent3.y)]).to(device)
        obs_2 = torch.FloatTensor([A[1][0]*(Agent2.x - Agent1.x), A[1][0]*(Agent2.y - Agent1.y), A[1][2]*(Agent2.x - Agent3.x), A[1][2]*(Agent2.y - Agent3.y)]).to(device)
        obs_3 = torch.FloatTensor([A[2][0]*(Agent3.x - Agent1.x), A[2][0]*(Agent3.y - Agent1.y), A[2][1]*(Agent3.x - Agent2.x), A[2][1]*(Agent3.y - Agent2.y)]).to(device)
         
    # Take an action for each agent    
        if np.random.random() > epsilon:
            action_1 = torch.argmax(net_1(obs_1))
            action_1 = action_1.cpu().numpy()
        else:
            action_1 = np.random.randint(0,8)    
        if np.random.random() > epsilon:
            action_2 = torch.argmax(net_2(obs_2))
            action_2 = action_2.cpu().numpy()
        else:
            action_2 = np.random.randint(0,8)
            
        if np.random.random() > epsilon:
            action_3 = torch.argmax(net_3(obs_3))
            action_3 = action_3.cpu().numpy()
        else:
            action_3 = np.random.randint(0,8)
        
        Agent1.action(action_1)
        Agent2.action(action_2)
        Agent3.action(action_3)
        
        laction_1 = [action_1]
        laction_2 = [action_2]
        laction_3 = [action_3]
        
        r_1 = 1 - A[0][1]*np.sqrt(np.power(Agent1.x-Agent2.x,2)+np.power(Agent1.y-Agent2.y,2)) - A[0][2]*np.sqrt(np.power(Agent1.x-Agent3.x,2)+np.power(Agent1.y-Agent3.y,2))
        r_2 = 1 - A[1][0]*np.sqrt(np.power(Agent2.x-Agent1.x,2)+np.power(Agent2.y-Agent1.y,2)) - A[1][2]*np.sqrt(np.power(Agent2.x-Agent3.x,2)+np.power(Agent2.y-Agent3.y,2))
        r_3 = 1 - A[2][0]*np.sqrt(np.power(Agent3.x-Agent1.x,2)+np.power(Agent3.y-Agent1.y,2)) - A[2][1]*np.sqrt(np.power(Agent3.x-Agent2.x,2)+np.power(Agent3.y-Agent2.y,2))
        
        new_obs_1 = torch.FloatTensor([A[0][1]*(Agent1.x - Agent2.x), A[0][1]*(Agent1.y - Agent2.y), A[0][2]*(Agent1.x - Agent3.x), A[0][2]*(Agent1.y - Agent3.y)]).to(device)
        new_obs_2 = torch.FloatTensor([A[1][0]*(Agent2.x - Agent1.x), A[1][0]*(Agent2.y - Agent1.y), A[1][2]*(Agent2.x - Agent3.x), A[1][2]*(Agent2.y - Agent3.y)]).to(device)
        new_obs_3 = torch.FloatTensor([A[2][0]*(Agent3.x - Agent1.x), A[2][0]*(Agent3.y - Agent1.y), A[2][1]*(Agent3.x - Agent2.x), A[2][1]*(Agent3.y - Agent2.y)]).to(device)
        
        future_qs_1 = net_1(new_obs_1)
        future_qs_2 = net_2(new_obs_2)
        future_qs_3 = net_3(new_obs_3)
        
        current_qs_1 = net_1(obs_1)
        current_qs_2 = net_2(obs_2)
        current_qs_3 = net_3(obs_3)
        
        max_future_q_1 = torch.argmax(future_qs_1)
        max_future_q_2 = torch.argmax(future_qs_2)
        max_future_q_3 = torch.argmax(future_qs_3)
        
        reward_1 = r_1 + A[0][1]*r_2 + A[0][2]*r_3
        reward_2 = A[1][0]*r_1 + r_2 + A[1][2]*r_3
        reward_3 = A[2][0]*r_1 + A[2][1]*r_2 + r_3
        
        max_future_sum_q_1 = max_future_q_1 + A[0][1]*max_future_q_2 + A[0][2]*max_future_q_3
        max_future_sum_q_2 = A[1][0]*max_future_q_1 + max_future_q_2 + A[1][2]*max_future_q_3
        max_future_sum_q_3 = A[2][0]*max_future_q_1 + A[2][1]*max_future_q_2 + max_future_q_3
        
        new_q_1 = reward_1 + DISCOUNT * max_future_sum_q_1
        new_q_2 = reward_2 + DISCOUNT * max_future_sum_q_2
        new_q_3 = reward_3 + DISCOUNT * max_future_sum_q_3
        
        output_1 = current_qs_1
        output_2 = current_qs_2
        output_3 = current_qs_3
        
        current_qs_1[action_1] = new_q_1
        current_qs_2[action_2] = new_q_2
        current_qs_3[action_3] = new_q_3
        
        net_1.train(output_1, current_qs_1)
        net_2.train(output_2, current_qs_2)
        net_3.train(output_3, current_qs_3)
        
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[Agent1.y][Agent1.x] = d[AGENT_N]
            env[Agent2.y][Agent2.x] = d[AGENT_N]
            env[Agent3.y][Agent3.x] = d[AGENT_N]
            
            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("", np.array(img))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        episode_reward_1 += r_1
        episode_reward_2 += r_2
        episode_reward_3 += r_3
        
        episode_rewards_1.append(episode_reward_1)
        episode_rewards_2.append(episode_reward_2)
        episode_rewards_3.append(episode_reward_3)
        
        epsilon *= EPSILON_DECAY
    print(f"{episode}: Rewards {episode_reward_1} {episode_reward_2} {episode_reward_3}")
 

# Need to Rethink Plotting?
 
# Get a moving average of episode rewards      
moving_avg_1 = np.convolve(episode_rewards_1, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
moving_avg_2 = np.convolve(episode_rewards_2, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
moving_avg_3 = np.convolve(episode_rewards_3, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

# Plot Average Rewards
plt.plot([i for i in range(len(moving_avg_1))], moving_avg_1)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel(f"episode #")
plt.show()
plt.plot([i for i in range(len(moving_avg_2))], moving_avg_2)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel(f"episode #")
plt.show()
plt.plot([i for i in range(len(moving_avg_3))], moving_avg_3)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel(f"episode #")
plt.show()
