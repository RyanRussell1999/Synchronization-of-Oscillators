#   Global Multiagent Q-Learning - 3 blobs
#   -------------------------------------------
#   Author: Ryan Russell

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
EPISODES = 5_000
SHOW_EVERY = 500
STEPS = 50
MOVE_PENALTY = 1
epsilon = 0.9
EPSILON_DECAY = 0.9998


start_q_table_1 = None # or filename
start_q_table_2 = None # or filename
start_q_table_3 = None # or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

AGENT_N = 1
d = {1: (255, 175, 0)}


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


# Initialize/Load Q-Tables
if start_q_table_1 is None:
    q_table_1 = np.random.uniform(low=0,high=1,size=(19,19,19,19,9))
else:
    with open(start_q_table_1, "rb") as f:
        q_table_1 = pickle.load(f)
 
if start_q_table_2 is None:
    q_table_2 = np.random.uniform(low=0,high=1,size=(19,19,19,19,9))
else:
    with open(start_q_table_2, "rb") as f:
        q_table_2 = pickle.load(f)
        
if start_q_table_3 is None:
    q_table_3 = np.random.uniform(low=0,high=1,size=(19,19,19,19,9))
else:
    with open(start_q_table_3, "rb") as f:
        q_table_3 = pickle.load(f)

episode_rewards_1 = [] 
episode_rewards_2 = [] 
episode_rewards_3 = []

video = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'XVID'), 24, (300,300))
 
# Q-Learning Algorithm 
for episode in range(EPISODES):
    Agent1 = Blob()
    Agent2 = Blob()
    Agent3 = Blob()
    
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
    
    for i in range(STEPS):
    # Get Distance apart from each agent for every agent
        obs_1 = [Agent1.x - Agent2.x, Agent1.y - Agent2.y, Agent1.x - Agent3.x, Agent1.y - Agent3.y]
        obs_2 = [Agent2.x - Agent1.x, Agent2.y - Agent1.y, Agent2.x - Agent3.x, Agent2.y - Agent3.y]
        obs_3 = [Agent3.x - Agent1.x, Agent3.y - Agent1.y, Agent3.x - Agent2.x, Agent3.y - Agent2.y]
       
        
    # Take an action for each agent    
        if np.random.random() > epsilon:
            a_1 = np.unravel_index(np.argmax(q_table_1[obs_1]), q_table_1.shape)
            action_1 = a_1[4]
        else: 
            action_1 = np.random.randint(0,8)
            
        if np.random.random() > epsilon:
            a_2 = np.unravel_index(np.argmax(q_table_2[obs_2]), q_table_2.shape)
            action_2 = a_2[4]
        else: 
            action_2 = np.random.randint(0,8)
            
        if np.random.random() > epsilon:
            a_3 = np.unravel_index(np.argmax(q_table_1[obs_1]), q_table_1.shape)
            action_3 = a_3[4]
        else: 
            action_3 = np.random.randint(0,8)
            
        Agent1.action(action_1)
        Agent2.action(action_2)
        Agent3.action(action_3)
        
        laction_1 = [action_1]
        laction_2 = [action_2]
        laction_3 = [action_3]
        
        r_1 = 1 - np.sqrt(np.power(Agent1.x-Agent2.x,2)+np.power(Agent1.y-Agent2.y,2)) - np.sqrt(np.power(Agent1.x-Agent3.x,2)+np.power(Agent1.y-Agent3.y,2))
        r_2 = 1 - np.sqrt(np.power(Agent2.x-Agent1.x,2)+np.power(Agent2.y-Agent1.y,2)) - np.sqrt(np.power(Agent2.x-Agent3.x,2)+np.power(Agent2.y-Agent3.y,2))
        r_3 = 1 - np.sqrt(np.power(Agent3.x-Agent1.x,2)+np.power(Agent3.y-Agent1.y,2)) - np.sqrt(np.power(Agent3.x-Agent2.x,2)+np.power(Agent3.y-Agent2.y,2))
        
        new_obs_1 = [Agent1.x - Agent2.x, Agent1.y - Agent2.y, Agent1.x - Agent3.x, Agent1.y - Agent3.y]
        new_obs_2 = [Agent2.x - Agent1.x, Agent2.y - Agent1.y, Agent2.x - Agent3.x, Agent2.y - Agent3.y]
        new_obs_3 = [Agent3.x - Agent1.x, Agent3.y - Agent1.y, Agent3.x - Agent2.x, Agent3.y - Agent2.y]
        
        max_future_q_1 = np.max(q_table_1[new_obs_1])
        max_future_q_2 = np.max(q_table_2[new_obs_2])
        max_future_q_3 = np.max(q_table_3[new_obs_3])
        
        current_q_1 = q_table_1[obs_1 + laction_1] 
        current_q_2 = q_table_2[obs_2 + laction_2] 
        current_q_3 = q_table_3[obs_3 + laction_3] 
        
        reward_1 = r_1 + r_2 + r_3
        reward_2 = reward_1
        reward_3 = reward_1
        
        max_future_sum_q_1 = max_future_q_1 + max_future_q_2 + max_future_q_3
        max_future_sum_q_2 = max_future_sum_q_1
        max_future_sum_q_3 = max_future_sum_q_1
        
        new_q_1 = (1-LEARNING_RATE) * current_q_1 + LEARNING_RATE * (reward_1 + DISCOUNT * max_future_sum_q_1)
        new_q_2 = (1-LEARNING_RATE) * current_q_2 + LEARNING_RATE * (reward_2 + DISCOUNT * max_future_sum_q_2)
        new_q_3 = (1-LEARNING_RATE) * current_q_3 + LEARNING_RATE * (reward_3 + DISCOUNT * max_future_sum_q_3)
        
        q_table_1[obs_1 + laction_1] = new_q_1
        q_table_2[obs_2 + laction_2] = new_q_2
        q_table_3[obs_3 + laction_3] = new_q_3
        
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[Agent1.y][Agent1.x] = d[AGENT_N]
            env[Agent2.y][Agent2.x] = d[AGENT_N]
            env[Agent3.y][Agent3.x] = d[AGENT_N]
            
            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("", np.array(img))
            image = np.array(img) 
            video.write(image)
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

# Get moving average     
moving_avg_1 = np.convolve(episode_rewards_1, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
moving_avg_2 = np.convolve(episode_rewards_2, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
moving_avg_3 = np.convolve(episode_rewards_3, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

# Plot Average of Episode Rewards
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

# Save Q-Tables
with open(f"qtable_1-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table_1, f)
with open(f"qtable_2-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table_2, f)
with open(f"qtable_3-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table_3, f)
