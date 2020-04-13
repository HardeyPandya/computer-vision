import cv2

#loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
custom_cascade = cv2.CascadeClassifier("data/cascade.xml")

#.detectMultiScale(image=gray, scalefactor=1.2, minNeighbors=5, flags=0, minSize=20, maxSize=400)
# scalefactor is parameter specifying how much the image size is reduced at each image scale

#Custom
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
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)

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
    #return frame

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectCustom(gray, frame)
    #detectFace(gray, frame)
    cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

video_capture.release()
cv2.destroyAllWindows()

