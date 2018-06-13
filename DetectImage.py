import threading
import queue
from PIL import Image
import time

class DetectImageThread(threading.Thread):
    def __init__(self, yolo, encoder, inputQueue, outputQueue = queue.Queue(maxsize=5)):
        super(DetectImageThread,self).__init__()
        self.queue = outputQueue
        self.imageQueue = inputQueue
        self.yolo = yolo
        self.encoder = encoder
        self.end = False
        self.setDaemon(True)
    def run(self):
        while True:
            if self.end:
                break
            if self.imageQueue.empty():
                time.sleep(0.001)
                continue
            imageZip = self.imageQueue.get()
            videoid, frameIndex, frame = imageZip
            boxs = self.yolo.detect_image(Image.fromarray(frame))
            features = self.encoder(frame,boxs)
            self.queue.put((videoid, frameIndex, frame, boxs, features))
    def stop(self):
        self.end = True