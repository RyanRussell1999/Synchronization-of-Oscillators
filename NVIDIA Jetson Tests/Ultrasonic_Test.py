import Jetson.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)

TRIG = 16
ECHO = 15

print "Distance Measurement In Progress"

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
print "Waiting For Sensor to Settle"
time.sleep(2)

start_time = time.time()
duration = 0

while duration < 10:
	GPIO.output(TRIG, True)
	time.sleep(0.00001)
	GPIO.output(TRIG, False)

	while GPIO.input(ECHO) == 0:
		pulse_start = time.time()

	while GPIO.input(ECHO) ==1:
		pulse_end = time.time()

	pulse_duration = pulse_end - pulse_start

	distance = (pulse_duration * 17150)/100

	distance = round(distance, 4)

	print "Distance:", distance, "m"

	duration = time.time() - start_time

	print(duration)

GPIO.cleanup()
