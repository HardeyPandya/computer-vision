import cv2
import Queue
import threading
import time
from cascades_utils import *
from threading_utils import *

#########################
cap = VideoCapture(0)
#cap = cv2.VideoCapture(0)
classifier = Cascade()

while True:
    frame = cap.read()#[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    classifier.detectCustom(gray, frame)
    classifier.detectFace(gray, frame)
    cv2.imshow("Video", frame)
    time.sleep(1)   # simulate time between events
    #if cv2.waitKey(33) & 0xFF == ord('q'): break
    if chr(cv2.waitKey(50)&255) == 'q':
        break

cap.release()
cv2.destroyAllWindows()
print("end")
#########################

'''
cd computer-vision/nxt-python/
conda activate nxt
python cascades_webcam.py
'''