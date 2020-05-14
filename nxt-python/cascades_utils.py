import cv2

class Cascade:
    def __init__(self, brick):
        self.face   = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
        self.eye    = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")
        self.smile  = cv2.CascadeClassifier("./cascades/haarcascade_smile.xml")
        self.custom = cv2.CascadeClassifier("./cascades/haarcascade_drone_20.xml")
        self.brick = brick

    def track(self, objs, frame):
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
                self.brick.moveHor(xdiff)
                self.brick.moveVer(ydiff)
            else:
                self.brick.shoot()
                cv2.rectangle(frame, \
                    (xcenter-int(w/2),ycenter-int(h/2)), \
                    (xcenter+int(w/2),ycenter+int(h/2)), (0, 0, 255), 5)

    #Custom
    def detectCustom(self, gray, frame):
        objs = self.custom.detectMultiScale(gray, 1.1, 10, 0, (24,24), (600,600))
        for (x,y,w,h) in objs:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 5)
        self.track(objs, frame)

    #eye detection
    def detectEye(self, gray, frame):
        eyes = self.eye.detectMultiScale(gray, 1.7, 10)#scaling factor and n_neighbours
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)

    #smile detection
    def detectSmile(self, gray, frame):
        smiles = self.smile.detectMultiScale(gray, 1.7, 10)
        for (x,y,w,h) in smiles:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)

    #function that will detect
    def detectFace(self, gray, frame):
        faces = self.face.detectMultiScale(gray, 1.3, 5)
        #faces is a list of tuples x,y,w,h
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            self.detectEye(roi_gray, roi_color)
            self.detectSmile(roi_gray, roi_color)
        self.track(faces, frame)
        #return frame