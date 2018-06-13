#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from video import VideoReader
import math

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import colorsys
import random
warnings.filterwarnings('ignore')


def put_text_multiline(img, text, org, font, fontScale, thickness, lineType, leftOrigin=True, lineSpacing=5, lineColor=None, color=(0,0,0), alignRight=False):
    lines = text.split('\n')
    sizes = [cv2.getTextSize(line, font, fontScale, thickness)[0] for line in lines]
    maxWidth = max([size[0] for size in sizes])
    y = sizes[0][1] + 1
    if leftOrigin:
        x = 0
    else:
        x = -maxWidth

    for i in range(len(lines)):
        fontcolor = (0,0,0)
        if lineColor is None or len(lineColor) == 0:
            fontcolor = color
        else:
            fontcolor = lineColor[min(i, len(lineColor) - 1)]
        alignXOffset = 0
        if alignRight:
            alignXOffset = maxWidth - sizes[i][0]
        cv2.putText(img,lines[i],(org[0] + x + alignXOffset, org[1] + y), font, fontScale, fontcolor,thickness,lineType)
        y += sizes[i][1] + lineSpacing
    return img

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

ramdom_colors = random_colors(100)

# 为图片创建空白侧边栏
def create_subview(frame, subview):
    return np.concatenate((frame,subview), axis=1)

def get_focalpoint_from_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2),int(bbox[1] + (bbox[3] - bbox[1]) * 0.1))

def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = 5
    nms_max_overlap = 1.0

    # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=32)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    display_flag = False

    videoReader = VideoReader("./test0604s.mp4")

    const_frame = videoReader.const_frame

    # 帧间隔时间 毫秒
    frame_interval = 1000 / videoReader.fps

    if writeVideo_flag:
        w = int(videoReader.width) * 2
        h = int(videoReader.height)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        sourcefps = videoReader.fps
        out = cv2.VideoWriter('output.avi', fourcc, sourcefps, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    # 轨迹图
    trackmap = None
    trackstorage = {}

    fps = 0.0
    finash_frame = 0
    start_time = time.time()
    while True:
        t1 = time.time()

        ret, frame = videoReader.read()  # frame shape 640*480*3
        if ret == False:
            break
        source_frame = np.copy(frame)

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)

        features = encoder(frame,boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        if writeVideo_flag or display_flag:
            if trackmap is None:
                trackmap = np.zeros(frame.shape, dtype=frame.dtype)

            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update >1 :
                    continue 

                if track.track_id not in trackstorage:
                    trackstorage[track.track_id] = []
                bbox = track.to_tlbr()
                trackstorage[track.track_id].append({'box': bbox, 'time':time.time()})

            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update >1 :
                    continue

                bbox = trackstorage[track.track_id][-1]['box']
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                # draw track map

                color = ramdom_colors[track.track_id % 100]

                if len(trackstorage[track.track_id]) <= 1:
                    cv2.rectangle(trackmap,
                        get_focalpoint_from_bbox(bbox),
                        get_focalpoint_from_bbox(bbox), color, 2)
                else:
                    cv2.line(trackmap,
                        get_focalpoint_from_bbox(bbox),
                        get_focalpoint_from_bbox(trackstorage[track.track_id][-2]['box']), color, 2)

            speed_text = ''
            speed_text_color = []

            for track in tracker.tracks:
                if not track.is_confirmed():
                    break
                distance = 0
                for i in range(1, len(trackstorage[track.track_id])):
                    pointa = get_focalpoint_from_bbox(trackstorage[track.track_id][i - 1]['box'])
                    pointb = get_focalpoint_from_bbox(trackstorage[track.track_id][i]['box'])
                    x = pointa[0] - pointb[0]
                    y = pointa[1] - pointb[1]
                    distance += math.sqrt(x * x + y * y)
                paytime = trackstorage[track.track_id][-1]['time'] - trackstorage[track.track_id][0]['time']
                speed_text += '%d: %.2f px/sec\n' % (track.track_id, distance / paytime)
                speed_text_color.append(ramdom_colors[track.track_id % 100])

            textview = np.zeros(trackmap.shape, dtype=trackmap.dtype)

            put_text_multiline(textview,
                speed_text,
                (trackmap.shape[1],0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                cv2.LINE_AA,
                leftOrigin=False,
                lineColor=speed_text_color,
                alignRight=True
            )

            for det in detections:
                pass
                #bbox = det.to_tlbr()
                # 显示检测框
                # cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
            frame = create_subview(frame, cv2.addWeighted(textview, 1, cv2.addWeighted(trackmap, 1 ,source_frame, 0.3, 0), 1, 0))

            for c in range(3):
                trackmap[:, :, c] = np.where(
                    trackmap[:, :, c] >= 2,
                    trackmap[:, :, c] - 2,
                    trackmap[:, :, c]
                )
                trackmap[:, :, c] = np.where(
                    trackmap[:, :, c] < 2,
                    np.zeros((trackmap.shape[0], trackmap.shape[1]), dtype=np.uint8),
                    trackmap[:, :, c]
                )

        if display_flag:
            cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

        # Press Q to stop!
        if display_flag and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = ( fps + (1./(time.time()-t1)) ) / 2
        finash_frame += 1
        print("fps = %f"%(fps))
        print("progress = %d/%d" % (finash_frame, const_frame))
        RTS = (const_frame - finash_frame) / fps
        print("time left = %02d:%02d:%02d" % (RTS / 3600 % 60, RTS / 60 % 60, RTS % 60))

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

    print('end')
    take_time = time.time() - start_time
    print('takeTime = %02d:%02d:%02d' % (take_time / 3600, take_time / 60 % 60, take_time % 60))

if __name__ == '__main__':
    main(YOLO())
