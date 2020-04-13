#NXT------------------------------------------------------
import cv2
import nxt.locator
from nxt.motor import *
import time
brick = nxt.locator.find_one_brick()

def moveHor(steps):
    motorLeft = Motor(brick, PORT_A)
    if steps>0: 
        speed = +10
        #right
        brick.play_tone_and_wait(400.0,100)
        brick.play_tone_and_wait(600.0,100)
    else:
        speed = -10
        #left
        time.sleep(1)
        brick.play_tone_and_wait(600.0,100)
        brick.play_tone_and_wait(400.0,100)
    degrees = 30
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

#LISTENER------------------------------------------------------

from pynput.keyboard import Key, Listener

def on_press(key):
    print('{0} pressed'.format(
        key))

def on_release(key):
    print('{0} release'.format(
        key))
    if   key == Key.up:
        moveVer(-10)
    elif key == Key.down:
        moveVer(10)
    elif key == Key.left:
        moveHor(-10)
    elif key == Key.right:
        moveHor(10)
    elif key == Key.end:
        shoot()
    elif key == Key.esc or key == Key.c or key == Key.q :
        # Stop listener
        return False

#VIDEO------------------------------------------------------

#loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#action function
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

#function that will detect
def detectFace(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #faces is a list of tuples x,y,w,h
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
    #track(faces, frame)


#LISTENER-------------------------------------------------------------
'''
# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release,
        ) as listener:
    listener.join()
    videoDetection()
'''
#VIDEO------------------------------------------------------------------

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectFace(gray, frame)
    cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    with Listener(
        on_press=on_press,
        on_release=on_release,
        ) as listener:
        listener.join()

video_capture.release()
cv2.destroyAllWindows()

