import threading
import queue

class ReadImageThread(threading.Thread):
    def __init__(self, videoid, videoReader, outputQueue = queue.Queue(maxsize=110)):
        super(ReadImageThread,self).__init__()
        self.queue = outputQueue
        self.videoReader = videoReader
        self.videoid = 1
        self.setDaemon(True)
    def run(self):
        frameIndex = 0
        while True:
            ret, frame = self.videoReader.read()  # frame shape 640*480*3
            if ret == False:
                break
            self.queue.put((self.videoid, frameIndex, frame))
            frameIndex+=1