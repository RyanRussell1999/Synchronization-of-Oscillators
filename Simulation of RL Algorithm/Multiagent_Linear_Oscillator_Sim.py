# Consensus of Oscillators using Consensus-Based Reinforcement Learning Control Algorithm with Linear Parameterization of Q-Function (Algorithm 2)
# 
# Created for Physical Implementation of Consensus Algorithm with one virtual leader and two physical follower oscillators
# 
# Simulation uses integrator model:
# x_dot = velocity = u_i
# x = position = x + T*u_i
#
# Simulation
#
# Author: Ryan Russell

import numpy as np
import time
from scipy.io import savemat
import random

### PARAMETERS

# Initial Values / Initialize Variables
x_i = []
x_j = []
x_leader = []
u_i = []
u_j = []
u_leader = []

Omega = np.pi/50 # Frequenct of Leader Oscillator

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
    elif k > 20_000:
        done = 1
    else:
        done = 0
    
    return done

### FUNCTIONS

# Calculate Action
def action(x_i, x_j, x_leader):
    u = (x_j[0] - x_i[0]) + (x_leader[0] - x_i[0])
    if u > 1:
        u = 1
    elif u < -1:
        u = -1
    return u
    
# Apply Action in Simulation
def Apply_Agent_Action(x_k, u, dt):
    x_k1 = np.array([x_k[0] + dt * u, u]) # Integrator Model
    return x_k1 

### MAIN LOOP
done = False # Done Flag
k = 0 # Time Step
dt = 0.1 # Time Period

# Initial States
x_i.append(np.array([0.1, 0]))
x_leader.append(np.array([0, 0]))
x_j.append(np.array([-0.2, 0]))

while not done:
    print(k, x_i[k][0], x_j[k][0], x_leader[k][0])
    
    # Determine Action
    u_leader.append(.02 * np.sin(k * dt * Omega))
    u_i.append(action(x_i[k], x_j[k], x_leader[k])) # u_i(k)
    u_j.append(action(x_j[k], x_i[k], x_leader[k])) # u_j(k)
    
    # Apply Action
    x_i.append(Apply_Agent_Action(x_i[k], u_i[k], dt))  # x_i(k+1) 
    x_j.append(Apply_Agent_Action(x_j[k], u_j[k], dt))  # x_j(k+1) 
    x_leader.append(Apply_Agent_Action(x_leader[k], u_leader[k], dt)) # x_leader(k+1)
    
    # Check to see if done
    done = CheckIfDone(x_i[k], x_j[k], k)
        
    # Update Time Step
    k = k + 1
    
    
### SAVE RESULTS IN MAT FILE
results = {'x_i': x_i, 'x_j' : x_j, 'x_leader' : x_leader,'u_i': u_i, 'u_j': u_j, 'u_leader': u_leader}

savemat("Linear_Oscillator_Results_{time}.mat".format(time = time.time()), results)