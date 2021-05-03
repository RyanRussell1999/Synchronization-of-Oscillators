# Code to Record Distance Measurements at different PWMs for Conversion to Cart Velocity
#
# Author: Ryan Russell

import Jetson.GPIO as GPIO
import numpy as np
import time
from scipy.io import savemat
import random

### PARAMETERS

# GPIO Pin Definitions
TRIG = 16 # BOARD pin 7
ECHO = 15 # BOARD pin 8
MOTOR_FORWARD_PWM = 32 # BOARD pin 32
MOTOR_BACKWARD_PWM = 33 # BOARD pin 33

# Initial Values / Initialize Variables
distances = []
recorded_time = []
recorded_pwm = []

### FAILURE CONDITIONS

def CheckIfDone(distance):
    # Check if Distance < 0.3
    if distance < 0.3:
        done = True
    else:
        done = False
    return done

### FUNCTIONS

# Get Distance and Velocity from Ultrasonic Sensor
def Get_Dist():

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

  	return distance

# Apply PWM to Motors
def PWM_to_Motors(motor_pwm):
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
print("PWM Test Started")
current_pwm = 10
k = 0 # Time Step

while current_pwm <= 100:
    print("Current PWM:", current_pwm)
    done = False # Done Flag
    start_time = time.time()
    
    # Apply PWM
    PWM_to_Motors(current_pwm)

    while not done:

        # Get Distance Measurements
        distance.append(Get_Dist)

        # Record Time
        recorded_time.append(time.time() - start_time)
        
        # Current PWM
        recorded_pwm.append(current_pwm)

        # Check to see if done
        done = CheckIfDone(distance[k])
            
        # Update Time Step
        k = k + 1
    
    # Stop Motors and wait 5 seconds to reset
    PWM_to_Motors(0)
    print("Test Complete! Motor Asleep for 5 seconds")
    time.sleep(5)
    
    # Increment up the PWM by 5 each Run
    current_pwm = current_pwm + 5
    
    
### SAVE RESULTS IN MAT FILE
results = {'distance': distance_vec, 'time': recorded_time_vec, 'PWM': recorded_pwm_vec}

savemat("Oscillator_Results_{time}.mat".format(time = time.time()), results)

### Close Devices and GPIO
pwm_forward_pin.stop()                         # stop PWM
pwm_backward_pin.stop()   
GPIO.cleanup()                     # resets GPIO ports used back to input mode