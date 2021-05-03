# Consensus of Oscillators using Consensus-Based Reinforcement Learning Control Algorithm with Linear Parameterization of Q-Function (Algorithm 2)
# 
# Created for Physical Implementation of Consensus Algorithm with one virtual leader and two physical follower oscillators
# 
# Simulation uses integrator model:
#   x_dot = velocity = u_i
#   x = position = x + T*u_i
#
#
# Simulation 
#
#    Ouput Saved in a mat file
#
# Author: Ryan Russell

import numpy as np
import time
from scipy.io import savemat
import random

### PARAMETERS

# RL Parameters
Nu = np.array([0.8, 1])
Lambda = 0.1 * np.identity(2)
Gamma = np.array([[1, 0], [0, 0]])
LEARNING_RATE = 0.005

# Initial Values / Initialize Variables
x_i = []
x_j = []
x_leader = []
reward_i = []
reward_j = []
theta_i = []
theta_j = []
phi_i = []
phi_j = []
phi_i_predict = []
phi_j_predict = []
u_i = []
u_j = []
u_i_predict = []
u_j_predict = []
u_leader = []

theta_i.append(np.array([0.01, 0.01, 0.01])) # theta_i[0]
theta_j.append(np.array([0.01, 0.01, 0.01])) # theta_j[0]

Omega = np.pi/50 # Frequency of Leader Oscillator

MAX_CHANGE = 0.001 # Max Change in Velocity per timestep

### FAILURE CONDITIONS

def CheckIfDone(x_ik, x_jk, k):
    
    # Done if other cart done or position < -1.5 m or > 1.5 m
    if x_jk[0] < -2.5:
        done = 1
    elif x_jk[0] > 2.5:
        done = 1
    elif x_ik[0] < -2.5:
        done = 1
    elif x_ik[0] > 2.5:
        done = 1
    elif k > 50_000:
        done = 1
    else:
        done = 0
    
    return done

### FUNCTIONS

# Calculate Action
def action(theta_k, x_i, x_j, x_leader, Nu):
    exponent = -(np.power(np.linalg.norm(x_i - x_j), 2) + np.power(np.linalg.norm(x_i - x_leader), 2))/(np.power(Nu[0],2))
    num = theta_k[1] * np.exp(exponent)
    den = 2 * theta_k[2]
    summ = (x_j[0] - x_i[0]) + (x_j[1] - x_i[1]) + (x_leader[0] - x_i[0]) + (x_leader[1] - x_i[1])
    
    return (num/den) * summ # Eq (59)

# Phi
def Phi(x_i, x_j, x_leader, Nu, u_i):
    # Phi 1
    num_1 = -(np.power(np.linalg.norm(x_i - x_j), 2) + np.power(np.linalg.norm(x_i - x_leader), 2))
    den_1 = np.power(Nu[0],2)
    summ_1 = np.power(np.linalg.norm(x_i - x_j), 2) + np.power(np.linalg.norm(x_i - x_leader), 2)
    phi_1 = np.exp(num_1/den_1) * summ_1
    
    # Phi 2
    num_2 = -(np.power(np.linalg.norm(x_i - x_j), 2) + np.power(np.linalg.norm(x_i - x_leader), 2))
    den_2 = np.power(Nu[1],2)
    summ_2 = ((x_i[0] - x_j[0]) * u_i) + ((x_i[1] - x_j[1]) * u_i) + ((x_i[0] - x_leader[0]) * u_i) + ((x_i[1] - x_leader[1]) * u_i)
    phi_2 = np.exp(num_2/den_2) * summ_2
    
    # Phi 3
    phi_3 = np.power(u_i, 2)
    
    return np.array([phi_1, phi_2, phi_3]) # Eq (58)
 
# Get Agent Reward
def Get_Reward(x_ik, x_jk, x_leaderk, u_ik, Gamma, Lambda):
    
    return np.matmul(np.matmul(np.transpose(x_ik - x_jk),Gamma), (x_ik - x_jk)) + np.matmul(np.matmul(np.transpose(x_ik - x_leaderk), Gamma), (x_ik - x_leaderk)) + np.transpose(u_ik) * Lambda[0][0] * u_ik
    
# Update Theta
def Theta_Update(theta_ik, theta_jk, learning_rate, reward_i, reward_j, phi_ikplus1, phi_ik, phi_jkplus1, phi_jk):

    return theta_ik + (learning_rate * ((reward_i + reward_j) + ((np.matmul(np.transpose(phi_ikplus1), theta_ik)) + (np.matmul(np.transpose(phi_jkplus1), theta_jk))) - ((np.matmul(np.transpose(phi_ik), theta_ik)) + (np.matmul(np.transpose(phi_ik),theta_ik)))) * phi_ik) # Eq (30)
    
# Apply Action in Simulation
def Apply_Agent_Action(x_k, u, dt):
    x_k1 = np.array([x_k[0] + dt * u, u]) # Integrator Model
    return x_k1 

### MAIN LOOP
done = False # Done Flag
k = 0 # Time Step
start_time = time.time()

# Get Initial States
x_i.append(np.array([0.3, 0]))
x_leader.append(np.array([0, 0]))
x_j.append(np.array([-0.4, 0]))

prev_time = time.time()

while not done:
    print(k, x_i[k][0], x_j[k][0], x_leader[k][0])
    
    # Determine Action
    dt = 0.1
    u_leader.append(.02 * np.sin(k * dt * Omega)) # u_leader(k)
    
    # Limit Control Input Change
    u_i_temp = action(theta_i[k], x_i[k], x_j[k], x_leader[k], Nu)
    u_j_temp = action(theta_j[k], x_j[k], x_i[k], x_leader[k], Nu)
    if k is 0:
        if abs(u_i_temp) > MAX_CHANGE:
            u_i_temp = np.sign(u_i_temp) * (MAX_CHANGE)
    elif abs(u_i_temp) > abs(u_i[k-1]) + MAX_CHANGE:
        u_i_temp = np.sign(u_i_temp-u_i[k-1]) * MAX_CHANGE + u_i[k-1]
        
    if k is 0:
        if abs(u_j_temp) > MAX_CHANGE:
            u_j_temp = np.sign(u_j_temp) * (MAX_CHANGE)
    elif abs(u_j_temp) > abs(u_j[k-1]) + MAX_CHANGE:
        u_j_temp = np.sign(u_j_temp-u_j[k-1]) * MAX_CHANGE + u_j[k-1]
    
    u_i.append(u_i_temp) # u_i(k)
    u_j.append(u_j_temp) # u_j(k)
    
    # Get Phi
    phi_i.append(Phi(x_i[k], x_j[k], x_leader[k], Nu, u_i[k])) # phi_i(k)
    phi_j.append(Phi(x_j[k], x_i[k], x_leader[k], Nu, u_j[k])) # phi_j(k)
    
    # Apply Action
    x_i.append(Apply_Agent_Action(x_i[k], u_i[k], dt))  # x_i(k+1) 
    x_j.append(Apply_Agent_Action(x_j[k], u_j[k], dt))  # x_j(k+1) 
    x_leader.append(Apply_Agent_Action(x_leader[k], u_leader[k], dt)) # x_leader(k+1)
    
    # Get Reward
    reward_i.append(Get_Reward(x_i[k], x_j[k], x_leader[k], u_i[k], Gamma, Lambda)) # r_i(k)
    reward_j.append(Get_Reward(x_j[k], x_i[k], x_leader[k], u_j[k], Gamma, Lambda)) # r_j(k)
    
    # Predicted next Action
    u_i_predict.append(action(theta_i[k], x_i[k+1], x_j[k+1], x_leader[k+1], Nu)) # u_i_predict(k)
    u_j_predict.append(action(theta_j[k], x_j[k+1], x_i[k+1], x_leader[k+1], Nu)) # u_j_predict(k)
    
    # Get Updated Phi
    phi_i_predict.append(Phi(x_i[k + 1], x_j[k + 1], x_leader[k+1], Nu, u_i_predict[k])) # phi_i_predict(k)
    phi_j_predict.append(Phi(x_j[k + 1], x_i[k + 1], x_leader[k+1], Nu, u_j_predict[k])) # phi_j_predict(k)
    
    # Update NN Parameters
    theta_i.append(Theta_Update(theta_i[k], theta_j[k], LEARNING_RATE, reward_i[k], reward_j[k], phi_i_predict[k], phi_i[k], phi_j_predict[k], phi_j[k])) # theta_i(k+1)
    theta_j.append(Theta_Update(theta_j[k], theta_i[k], LEARNING_RATE, reward_j[k], reward_i[k], phi_j_predict[k], phi_j[k], phi_i_predict[k], phi_i[k])) # theta_j(k+1)
    
    # Check to see if done
    done = CheckIfDone(x_i[k], x_j[k], k)
        
    # Update Time Step
    k = k + 1
    
    
### SAVE RESULTS IN MAT FILE
results = {'x_i': x_i, 'x_j' : x_j, 'x_leader' : x_leader,'u_i': u_i, 'u_j': u_j, 'reward_i': reward_i, 'reward_j': reward_j,'theta_i': theta_i, 'theta_j': theta_j}

savemat("Oscillator_Results_{time}.mat".format(time = time.time()), results)