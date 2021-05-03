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


dc = 0
pwm_foreword.start(dc)
pwm_backward.start(dc)


#XBee function for callback recieving messages
def my_data_received_callback(xbee_message):
    address = xbee_message.remote_device.get_64bit_addr()
    
    global data_recieved
    global other_dc
    
    data = xbee_message.data.decode()
    other_dc = np.fromstring(data[1:-1], sep=',')
    
    data_recieved = True

# Setup Devices

local_device = XBeeDevice("/dev/ttyUSB0", 9600)
local_device.open()
remote_device = RemoteXBeeDevice(local_device, XBee64BitAddress.from_hex_string("0013A20041C80821"))
local_device.add_data_received_callback(my_data_received_callback)

# Apply Inital Motor PWM
motor_dc = 80
pwm_foreword.ChangeDutyCycle(motor_dc)
print("Initial PWM:", motor_dc)
data_recieved = False

# Run Motor for 5 seconds
start_time = time.time()
run_motor_time = time.time() - start_time
while run_motor_time < 5:
	run_motor_time = time.time() - start_time

# Send Data
dc = np.array([motor_dc])
data = np.array2string(dc, precision=2, separator=',',
                      suppress_small=True)
local_device.send_data(remote_device, data)

while data_recieved is False:
    time_dc = time.time()
     
#define avg. Duty Cycle  between both Carts
new_dc = (other_dc + dc) / 2

# Apply new PWM
pwm_foreword.ChangeDutyCycle(new_dc)
print("Resulting PWM:", new_dc)
time.sleep(5) # Run for 5 seconds

print("Program Ending")

pwm_foreword.stop()
pwm_backward.stop()
local_device.close()
GPIO.cleanup()
