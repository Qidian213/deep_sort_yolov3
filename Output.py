import threading
import queue
from deep_sort.detection import Detection
import numpy as np
from deep_sort import preprocessing
import time
import cv2
from tools.tracker_results_painter import TrackerResultsPainter

class OutputThread(threading.Thread):
    def __init__(self, inputQueue, tracker, video_output_filename='output.avi', fourcc = cv2.VideoWriter_fourcc(*'MJPG'), fps = 10, nms_max_overlap = 1.0):
        super(OutputThread,self).__init__()
        self.queue = inputQueue
        self.setDaemon(True)
        self.nms_max_overlap = nms_max_overlap
        self.tracker = tracker
        self.end = False
        self.video_output_filename = video_output_filename
        self.fourcc = fourcc
        self.fps = fps
        self.video_writer = None

        self.current_frame_index = 0
        # 由于输入队列中数据顺序混乱 先存放在数据池中 处理后按顺序输出
        self.dataPool = {}

        self.painter = TrackerResultsPainter(self.tracker)
    def run(self):
        while True:
            if self.end:
                break
            if self.queue.empty():
                time.sleep(0.001)
                continue
            self._putdata(self.queue.get())

    def _putdata(self, data):
        videoid, frameIndex, frame, boxs, features = data
        self.dataPool[frameIndex] = data
        while self.current_frame_index in self.dataPool:
            self._workdata(self.dataPool[self.current_frame_index])
            del self.dataPool[self.current_frame_index]
            self.current_frame_index += 1

    def _workdata(self, data):
        videoid, frameIndex, frame, boxs, features = data

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        result = self.painter.draw(frame)

        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(self.video_output_filename, self.fourcc, self.fps, (result.shape[1],result.shape[0]))

        self.video_writer.write(result)

    def stop(self):
        self.end = True