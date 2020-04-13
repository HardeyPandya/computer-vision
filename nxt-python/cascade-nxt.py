import cv2
import nxt.locator
from nxt.motor import *
import time

def moveHor(steps):
	motorLeft = Motor(brick, PORT_A)
	if steps>0: 
		speed = +20
		#right
		brick.play_tone_and_wait(400.0,100)
		brick.play_tone_and_wait(600.0,100)
	else:
		speed = -20
		#left
		time.sleep(1)
		brick.play_tone_and_wait(600.0,100)
		brick.play_tone_and_wait(400.0,100)
	degrees = 50
	motorLeft.turn(speed, degrees)

def moveVer(steps):
	motorRight = Motor(brick, PORT_B)
	if steps<0: 
		speed = +50
		#up
		brick.play_tone_and_wait(800.0,100)
		brick.play_tone_and_wait(1000.0,100)
	else:
		speed = -50
		#down
		brick.play_tone_and_wait(1000.0,100)
		brick.play_tone_and_wait(800.0,100)
	degrees = 50
	motorRight.turn(speed, degrees)	

def shoot():
	brick.play_tone_and_wait(300.0,100)
	time.sleep(0.1)
	brick.play_tone_and_wait(100.0,100)

	motorTrigger = Motor(brick, PORT_C)
	motorTrigger.turn(-50, 50)
	motorTrigger.turn( 50, 50)

brick = nxt.locator.find_one_brick()
video_capture = cv2.VideoCapture(0)

while True:
	#time.sleep(1)
	_, frame = video_capture.read()
	cv2.imshow("Video",frame)
	if cv2.waitKey(1) & 0xFF == ord('a'):
		moveHor(-10)
	if cv2.waitKey(1) & 0xFF == ord('d'):
		moveHor(10)
	if cv2.waitKey(1) & 0xFF == ord('w'):
		moveVer(10)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		moveVer(-10)
	if cv2.waitKey(1) & 0xFF == ord('e'):
		shoot()

	if cv2.waitKey(1) & 0xFF == ord('q'): break

video_capture.release()
cv2.destroyAllWindows()
