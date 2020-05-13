import cv2
import Queue
import threading
import time
from cascades_utils import *
from threading_utils import *
from nxt_utils import *



brick = Mindstorms()

cap = VideoCapture("/dev/video2")

classifier = Cascade(brick)


while True:
    frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    classifier.detectCustom(gray, frame)
    #classifier.detectFace(gray, frame)

    cv2.imshow("Video", frame)
    '''
    if cv2.waitKey(1) & 0xFF == ord('a'):
        brick.moveHor(-10)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        brick.moveHor(10)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        brick.moveVer(10)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        brick.moveVer(-10)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        brick.shoot()
    '''

    if chr(cv2.waitKey(50)&255) == 'q':break

cap.release()
cv2.destroyAllWindows()
print("end")
#########################

'''
cd computer-vision/nxt-python/
conda activate nxt
python nxt_webcam.py

'''