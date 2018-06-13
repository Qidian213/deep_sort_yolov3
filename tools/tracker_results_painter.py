import cv2
import colorsys
import random
import numpy as np

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple(map(lambda c: int(c * 255) ,colorsys.hsv_to_rgb(*c))), hsv))
    random.shuffle(colors)
    return colors

# 为图片创建空白侧边栏
def create_subview(frame, subview):
    return np.concatenate((frame,subview), axis=1)

def get_focalpoint_from_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2),int(bbox[1] + (bbox[3] - bbox[1]) * 0.1))

ramdom_colors = random_colors(100)

class TrackerResultsPainter:
    def __init__(self, tracker):
        self.tracker = tracker
        self.trackmap = None

    def draw(self, source_frame):
        frame = np.copy(source_frame)

        if self.trackmap is None:
            self.trackmap = np.zeros(frame.shape, dtype=frame.dtype)

        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue

            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            # draw track map

            color = ramdom_colors[track.track_id % 100]

            if track.prev_mean is None:
                cv2.rectangle(self.trackmap,
                    get_focalpoint_from_bbox(bbox),
                    get_focalpoint_from_bbox(bbox), color, 2)
            else:
                cv2.line(self.trackmap,
                    get_focalpoint_from_bbox(bbox),
                    get_focalpoint_from_bbox(track.to_tlbr(track.prev_mean)), color, 2)

        frame = create_subview(frame, cv2.addWeighted(self.trackmap, 1 ,source_frame, 0.3, 0))

        for c in range(3):
            self.trackmap[:, :, c] = np.where(
                self.trackmap[:, :, c] >= 2,
                self.trackmap[:, :, c] - 2,
                self.trackmap[:, :, c]
            )
            self.trackmap[:, :, c] = np.where(
                self.trackmap[:, :, c] < 2,
                np.zeros((self.trackmap.shape[0], self.trackmap.shape[1]), dtype=np.uint8),
                self.trackmap[:, :, c]
            )

        return frame