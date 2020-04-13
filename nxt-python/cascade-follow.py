import cv2
import nxt.locator
from nxt.motor import *
import time

#loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
custom_cascade = cv2.CascadeClassifier("cascade.xml")

#.detectMultiScale(image=gray, scalefactor=1.2, minNeighbors=5, flags=0, minSize=20, maxSize=400)
# scalefactor is parameter specifying how much the image size is reduced at each image scale

#Custom

def moveHor(steps):
	motorLeft = Motor(brick, PORT_A)
	if steps>0: 
		speed = +50
		#right
		brick.play_tone_and_wait(400.0,100)
		brick.play_tone_and_wait(600.0,100)
	else:
		speed = -50
		#left
		time.sleep(1)
		brick.play_tone_and_wait(600.0,100)
		brick.play_tone_and_wait(400.0,100)
	degrees = 100
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
	degrees = 100
	motorRight.turn(speed, degrees)	

def shoot():
	brick.play_tone_and_wait(300.0,100)
	time.sleep(0.1)
	brick.play_tone_and_wait(100.0,100)

	motorTrigger = Motor(brick, PORT_C)
	motorTrigger.turn(-50, 50)
	motorTrigger.turn( 50, 50)

def track(objs, frame):
	(ymax, xmax) = frame.shape[:2]
	ycenter = int(ymax/2)
	xcenter = int(xmax/2)
	for (x,y,w,h) in objs:
		ydiff = y-ycenter+int(h/2)
		xdiff = x-xcenter+int(w/2)
		coordinates = "y: {} x: {}".format(ydiff, xdiff)
		print coordinates
		cv2.putText(frame, coordinates, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)


		if not(abs(ydiff)<=h/2 and abs(xdiff)<=w/2):
			moveHor(xdiff)
			moveVer(ydiff)
		else:
			shoot()
			cv2.rectangle(frame, \
				(xcenter-int(w/2),ycenter-int(h/2)), \
				(xcenter+int(w/2),ycenter+int(h/2)), (0, 0, 255), 5)

def detectCustom(gray, frame):
	objs = custom_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(24,24), maxSize=(600,600))
	for (x,y,w,h) in objs:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 5)

#eye detection
def detectEye(gray, frame):
	eyes = eye_cascade.detectMultiScale(gray, 1.7, 10)#scaling factor and n_neighbours
	for (x,y,w,h) in eyes:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)

#smile detection
def detectSmile(gray, frame):
	smiles = smile_cascade.detectMultiScale(gray, 1.7, 10)
	for (x,y,w,h) in smiles:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 1)

#function that will detect
def detectFace(gray, frame):
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	#faces is a list of tuples x,y,w,h
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		detectEye(roi_gray, roi_color)
		detectSmile(roi_gray, roi_color)
	track(faces, frame)

brick = nxt.locator.find_one_brick()

video_capture = cv2.VideoCapture(0)
#print video_capture.read()[1].shape
#(480, 640, 3)

while True:
	#time.sleep(1)
	_, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detectCustom(gray, frame)
	detectFace(gray, frame)
	cv2.imshow("Video",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): break

video_capture.release()
cv2.destroyAllWindows()

