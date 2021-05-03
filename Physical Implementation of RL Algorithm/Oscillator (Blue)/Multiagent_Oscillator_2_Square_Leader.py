# Consensus of Oscillators using Consensus-Based Reinforcement Learning Control Algorithm with Linear Parameterization of Q-Function (Algorithm 2)
# 
# Created for Physical Implementation of Consensus Algorithm with one virtual leader and one physical follower oscillator
# 
# Follower 2 Oscillator - Blue XBee
#
# Author: Ryan Russell

import Jetson.GPIO as GPIO
import numpy as np
import time
from scipy.io import savemat
import random

### PARAMETERS

# GPIO Pin Definitions
TRIG = 16 # BOARD pin 16
ECHO = 15 # BOARD pin 15
MOTOR_FORWARD_PWM = 32 # BOARD pin 32
MOTOR_BACKWARD_PWM = 33 # BOARD pin 33

# RL Parameters
Nu = np.array([0.8, 1])
Lambda = 0.1 * np.ones([2,2])
Gamma = np.array([[1, 0], [0, 0]])
LEARNING_RATE = 0.0001

# Initial Values / Initialize Variables
x_i = []
x_j = []
x_leader = []
reward = []
theta = []
phi = []
phi_j = []
phi_predict = []
u = []
u_leader = []
u_predict =[]
time_vec = []

theta.append(np.array([0.001, 0.001, 0.001])) # theta[0]

Leader_Control = 0.3

### FAILURE CONDITIONS

def CheckIfDone(x_ik):
    
    # Done if position < -1.5 m or > 1.5 m
    if x_ik[0] < -1.5:
        done = 1
    elif x_ik[0] > 1.5:
        done = 1
    else:
        done = 0
    
    return done

### FUNCTIONS

# Get Distance and Velocity from Ultrasonic Sensor
def Get_Dist_Vel(previous_distance, elapsed_time):

    # Send Trigger
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Read Echo
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) ==1:
        pulse_end = time.time()

    # Calculate Distance from Pulse Duration
    pulse_duration = pulse_end - pulse_start

    distance = (pulse_duration * 17150)/100 # m
    
    distance = distance - 2 # Offset the distance by 2 m for a range of -2 m to 2 m
    
    # Calculate average velocity over the time step
    if elapsed_time == 0:
        velocity = 0 # m/s
    else:
        velocity = (distance - previous_distance)/elapsed_time # m/s
    
    # Place results in a vector
    x_states = np.array([distance, velocity])

    return x_states

# Apply PWM to Motors
def PWM_to_Motors(motor_pwm):
	# Make 40 PWM the minimum to deliver to motors
    if abs(motor_pwm) < 15:
        motor_pwm = 0
    
    # Limit the PWM Duty Cycle
    if motor_pwm > 100:
        motor_pwm = 100
    if motor_pwm < -100:
        motor_pwm = -100

	# Apply Duty Cycle
    if motor_pwm > 0:
        pwm_forward_pin.ChangeDutyCycle(motor_pwm)
        pwm_backward_pin.ChangeDutyCycle(0)
    else:
        pwm_forward_pin.ChangeDutyCycle(0)
        pwm_backward_pin.ChangeDutyCycle(-motor_pwm)

# Calculate Action
def action(theta_k, x_i, x_leader, Nu):
    exponent = -np.power(np.linalg.norm(x_i - x_leader), 2)/(np.power(Nu[0],2))
    num = theta_k[1] * np.exp(exponent)
    den = 2 * theta_k[2]
    summ = (x_leader[0] - x_i[0]) + (x_leader[1] - x_i[1])
    
    return (num/den) * summ # Eq (59)

# Phi
def Phi(x_i, x_leader, Nu, u_i):
    # Phi 1
    num_1 = -np.power(np.linalg.norm(x_i - x_leader), 2)
    den_1 = np.power(Nu[0],2)
    summ_1 = np.power(np.linalg.norm(x_i - x_leader), 2)
    phi_1 = np.exp(num_1/den_1) * summ_1
    
    # Phi 2
    num_2 = -np.power(np.linalg.norm(x_i - x_leader), 2)
    den_2 = np.power(Nu[1],2)
    summ_2 = ((x_i[0] - x_leader[0]) * u_i) + ((x_i[1] - x_leader[1]) * u_i)
    phi_2 = np.exp(num_2/den_2) * summ_2
    
    # Phi 3
    phi_3 = np.power(u_i, 2)
    
    return np.array([phi_1, phi_2, phi_3]) # Eq (58)
 
# Get Agent Reward
def Get_Reward(x_ik, x_leaderk, u_ik, Gamma, Lambda):
    
    return np.matmul(np.matmul(np.transpose(x_ik - x_leaderk), Gamma), (x_ik - x_leaderk)) + np.transpose(u_ik) * Lambda[0][0] * u_ik
    
# Update Theta
def Theta_Update(theta_ik, learning_rate, reward_i, phi_ikplus1, phi_ik):

    return theta_ik + (learning_rate * (reward_i + ((np.matmul(np.transpose(phi_ikplus1), theta_ik)) - ((np.matmul(np.transpose(phi_ik), theta_ik)) + (np.matmul(np.transpose(phi_ik),theta_ik)))) * phi_ik)) # Eq (30)
    
# Apply Action in Simulation
def Apply_Leader_Agent_Action(x_k, u, dt):
    x_k1 = np.array([x_k[0] + dt * u, u]) # Integrator Model
    return x_k1
    
### GPIO SETUP
GPIO.setmode(GPIO.BOARD) 

# Motor Pin Setup
GPIO.setup(MOTOR_FORWARD_PWM, GPIO.OUT)  # Set GPIO pin 32 to output mode.
pwm_forward_pin = GPIO.PWM(MOTOR_FORWARD_PWM, 100)   # Initialize PWM on Forward Pin at 100Hz frequency
GPIO.setup(MOTOR_BACKWARD_PWM, GPIO.OUT)  # Set GPIO pin 33 to output mode.
pwm_backward_pin = GPIO.PWM(MOTOR_BACKWARD_PWM, 100)   # Initialize PWM on Backward Pin at 100Hz frequency

# Start PWM with 0% duty cycle
pwm_forward_pin.start(0)
pwm_backward_pin.start(0)

# Ultrasonic Sensor Pin Setup
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

### MAIN LOOP
done = False # Done Flag
k = 0 # Time Step
start_time = time.time()

# Get Initial States
x_i.append(Get_Dist_Vel(0, 0))
current_time = time.time() - start_time
x_leader.append([0.25,0]) # x_leader(0)
sim_start_time = time.time()

print("Program Starting")

while not done:

    print("x_i:", x_i[k], "x_leader:", x_leader[k])
    
    # Determine Action
    u.append(action(theta[k], x_i[k], x_leader[k], Nu)) # u(k)
    print(-np.sign(u[k]) * np.sqrt(abs(u[k]))/0.0135)

    # Get Phi
    phi.append(Phi(x_i[k], x_leader[k], Nu, u[k])) # phi(k)
    
    # Apply Action
    PWM_to_Motors(-np.sign(u[k]) * np.sqrt(abs(u[k]))/0.0135)
    time_action_applied = time.time() - sim_start_time
    time_vec.append(time_action_applied)
    
    # Updated States
    if time.time() - start_time < 1:
        time.sleep(1 - (time.time() - start_time))
    elapsed_time = time.time() - start_time  
    x_i.append(Get_Dist_Vel(x_i[k][0], elapsed_time)) # x_i(k+1)
    
    # Update Leader States
    if np.remainder(k,2) == 0:
        Leader_Control = -Leader_Control
    u_leader.append(Leader_Control) # u_leader(k)
    x_leader.append(Apply_Leader_Agent_Action(x_leader[k], u_leader[k], 1)) # x_leader(k+1)
    start_time = time.time()  
        
    # Get Reward
    reward.append(Get_Reward(x_i[k], x_leader[k], u[k], Gamma, Lambda)) # r(k)
    
    # Predicted next Action
    u_predict.append(action(theta[k], x_i[k+1], x_leader[k+1], Nu)) # u_predict(k)

    # Get Predicted Phi
    phi_predict.append(Phi(x_i[k + 1], x_leader[k+1], Nu, u_predict[k])) # phi_predict(k)
    
    # Update NN Parameters
    theta.append(Theta_Update(theta[k], LEARNING_RATE, reward[k], phi_predict[k], phi[k])) # theta(k+1)
    
    # Check to see if done
    done = CheckIfDone(x_i[k])
        
    # Update Time Step
    k = k + 1
    print(k)
    
### SAVE RESULTS IN MAT FILE  
results = {'x_i': x_i, 'x_leader': x_leader, 'u': u, 'reward_i': reward, 'theta_i': theta, 'time': time_vec}

savemat("Oscillator_Results_{time}.mat".format(time = time.time()), results)

### Close Devices and GPIO
pwm_forward_pin.stop()             # stop PWM
pwm_backward_pin.stop()             # stop PWM
GPIO.cleanup()                     # resets GPIO ports used back to input model
