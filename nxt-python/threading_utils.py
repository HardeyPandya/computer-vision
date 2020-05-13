import cv2
import Queue
import threading
import time

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue.Queue()
        self.writer = None
        #self.t = threading.Thread(target=self._reader)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        #self.t.start()

    def read(self):
        frame = self.q.get()
        return frame
    
    def release(self):
        self.writer.release()
        self.cap.release()
        self.q.join()
        #self.q.put(None)
        #self.t.join()
        print("all released")

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("break")
                #self.release()
                break
            if not self.q.empty():
                try:
                    dumped_frame = self.q.get()   # discard previous (unprocessed) frame
                    if self.writer != None:
                        #print("q written")
                        self.writer.write(dumped_frame)
                except Queue.Empty:
                    pass
            self.q.put(frame)
            self.q.task_done()
