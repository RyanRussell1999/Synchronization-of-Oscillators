# Match PWM for two Carts; Cart 1 code
#Feb. 15, 2021
#Jakob Harig
#Cart-Pole Project Senior Capstone


# cart 2 has orange Xbee

import Jetson.GPIO as GPIO #import the GPIO library
import time #import timing options
import numpy as np # for string to number conversion with XBee transmission 

from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice, XBee64BitAddress # for XBee communication 

GPIO.setmode(GPIO.BOARD) 

GPIO.setup(32, GPIO.OUT)  # Set GPIO pin 12 to output mode.
GPIO.setup(33, GPIO.OUT)  # Set GPIO pin 13 to output mode.
pwm_foreword = GPIO.PWM(32, 100)   # Initialize PWM  foreword on pwmPin 100Hz frequency
pwm_backward = GPIO.PWM(33, 100)   # Initialize PWM  foreword on pwmPin 100Hz frequency


# Pin Definitions for motor driver
motor_pin_a = 32  # BOARD pin 32
motor_pin_b = 33  # BOARD pin 33


#XBee function for callback recieving messages
def my_data_received_callback(xbee_message):
    address = xbee_message.remote_device.get_64bit_addr()
    
    global data 
    global dc
    
    data = xbee_message.data.decode()
    data = np.fromstring(data[1:-1], sep=',')
    
    data_recieved = True
    print(data)

    

#Cart 2 PWM information and specs.

dc = 0; #set inital pwm duty cycle to zero for initial part of program
pwm_foreword.start(dc) # start the pwm signal at inital duty cycle 
# Set both pins LOW to keep the motor idle
# You can keep one of them HIGH and the LOW to start with rotation in one direction 
    GPIO.setup(motor_pin_a, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(motor_pin_b, GPIO.OUT, initial=GPIO.LOW)
    
pwm_backward.start(dc)  # start the pwm backward signal at zero
# Set both pins LOW to keep the motor idle
# You can keep one of them HIGH and the LOW to start with rotation in one direction 
    GPIO.setup(motor_pin_a, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(motor_pin_b, GPIO.OUT, initial=GPIO.LOW)



try:
    while data_recieved = False:
    #drive the cart 2 foreword
    for dc in range(30) #set Cart 2 to duty cycle 3
    #need to set this to motor C to drive foreword
        GPIO.output(motor_pin_a, curr_value_pin_a)
        GPIO.output(motor_pin_b, curr_value_pin_b)
        curr_value_pin_a ^= GPIO.HIGH #make motor A HIGH to drive foreword
        curr_value_pin_b ^= GPIO.LOW #make motor B LOW to drive foreword
        
        pwm_foreword.ChangeDutyCycle(dc) #change pwm to desired duty cycle
        print("Outputting {} as duty cycle while motor {} is high and motor {} is low".format(dc,curr_value_pin_a, curr_value_pin_b))
        time.sleep(3) #drive cart 2 foreword for 3 seconds
    pwm_foreword.ChangeDutyCycle(0)
    
    #start to drive the Cart backwards
    for dc in range(30) #set Cart 2 to duty cycle 3
    #need to set this to motor B to drive backwards 
        GPIO.output(motor_pin_a, curr_value_pin_a)
        GPIO.output(motor_pin_b, curr_value_pin_b)
        curr_value_pin_a ^= GPIO.LOW #make motor A LOW to drive backward
        curr_value_pin_b ^= GPIO.HIGH #make motor B HIGH to drive backward
        
        pwm_backward.ChangeDutyCycle(dc) #change to desired pwm duty cycle
        print("Outputting {} as duty cycle while motor {} is high and motor {} is low".format(dc,curr_value_pin_a, curr_value_pin_b))	
        time.sleep(3) # drive cart 2 backwards for 3 seconds
    
    pwm_backward.ChangeDutyCycle(0)
    
    #XBee communication info
    local_device = XBeeDevice("/dev/ttyUSB0", 9600) #will probaly need to change COM# here
    local_device.open()
    
    #sending imformation for XBee
    dc_string = "35" #sending the string form of the pwm cycle 
    remote_device = RemoteXBeeDevice(local_device, XBee64BitAddress.from_hex_string("0013A20041C80821")) #specify MAC Adress for Xbee we are communicating with; Orange sending to Blue 
    local_device.send_data(remote_device, dc_string) #where you actually send the data to the other XBee 
    local_device.close() #close the device for sending
    
    
    
    # Setup Devices
    local_device = XBeeDevice("/dev/ttyUSB0", 9600)
    local_device.open()
    remote_device = RemoteXBeeDevice(local_device, XBee64BitAddress.from_hex_string("0013A20041C80821"))
    local_device.add_data_received_callback(my_data_received_callback)

    for i in range(100000):
        data = np.array([30])
        data = np.array2string(data, precision=2, separator=',',
                          suppress_small=True)
        local_device.send_data(remote_device, data)

    local_device.close()
  while data_recieved = True:  
    #define avg. Duty Cycle  between both Carts
    new_dc = (data + dc) / 2
    
    print("/n {}".format(new_dc))
    #setting new DC pwm after xbee message recieved 
    for new_dc in range(new_dc) #set Cart 2 to duty cycle avg. dc
    #need to set this to motor C to drive foreword
        GPIO.output(motor_pin_a, curr_value_pin_a)
        GPIO.output(motor_pin_b, curr_value_pin_b)
        curr_value_pin_a ^= GPIO.HIGH #make motor C HIGH to drive foreword
        curr_value_pin_b ^= GPIO.LOW #make motor D LOW to drive foreword
        
        pwm_foreword.ChangeDutyCycle(new_dc) #change pwm to desired duty cycle
        print("Outputting {} as duty cycle while motor {} is high and motor {} is low".format(new_dc,curr_value_pin_a, curr_value_pin_b)
        time.sleep(3) #drive cart 2 foreword for 3 seconds
    pwm_foreword.ChangeDutyCycle(0)
    for new_dc in range(new_dc) #set Cart 2 to duty cycle 35
    #need to set this to motor D to drive backwards 
        GPIO.output(motor_pin_a, curr_value_pin_a)
        GPIO.output(motor_pin_b, curr_value_pin_b)
        curr_value_pin_a ^= GPIO.LOW #make motor c LOW to drive backward
        curr_value_pin_b ^= GPIO.HIGH #make motor d HIGH to drive backward
        
        pwm_backward.ChangeDutyCycle(new_dc) #change to desired pwm duty cycle
        print("Outputting {} as duty cycle while motor {} is high and motor {} is low".format(new_dc,curr_value_pin_b, curr_value_pin_a))	
        time.sleep(3) # drive cart 2 backwards for 3 seconds
    pwm_backward.ChangeDutyCycle(0)
    

GPIO.cleanup