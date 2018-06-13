import threading
from video import video_synth_base
import cv2

class VideoReader:
    def __init__(self, path, cache_length=5):
        self.path = path
        self.cache_length = cache_length
        self.cache = []

        self._read_threading = threading.Thread(target=self._read_thread, name='readThread')
        self._read_event = threading.Event()
        self._pop_event = threading.Event()
        self._read_end = False
        self.video_capture = video_synth_base.create_capture(self.path)

        self._read_threading.setDaemon(True)
        self._read_threading.start()

    def _read_thread(self):
        while True:
            ret, frame = self.video_capture.read()
            if ret != True:
                self._read_end = True
                self.video_capture.release()
                break
            else:
                self.cache.append(frame)

            self._read_event.set()

            if len(self.cache) >= self.cache_length:
                self._pop_event.clear()
                self._pop_event.wait()
    
    def read(self):
        if len(self.cache) != 0:
            data = self.cache.pop(0)
            self._pop_event.set()
            return (True, data)
        elif self._read_end:
            return (False, None)
        else:
            self._read_event.clear()
            self._read_event.wait()
            return self.read()

    @property
    def width(self):
        return self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self):
        return self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self):
        return self.video_capture.get(cv2.CAP_PROP_FPS)

    @property
    def const_frame(self):
        return self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
