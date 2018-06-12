
# Introduction

  https://github.com/nwojke/deep_sort
  
  https://github.com/qqwweee/keras-yolo3

# Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO_DEEP_SORT 

```
   wget https://pjreddie.com/media/files/yolov3.weights
   python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
   python demo.py
```

# Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are needed to run the tracker:

    NumPy
    sklean
    OpenCV

Additionally, feature generation requires TensorFlow-1.4.0.

# Note 
 file model_data/mars-small128.pb  had convert to tensorflow-1.4.0
 
 file model_data/yolo.h5 is to large to upload ,so you need convert it from Darknet Yolo model to a keras model by yourself
 
 yolo.h5 model can download from https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing , use tensorflow1.4.0
 
# Test
 use : 'video_capture = cv2.VideoCapture('path to video')' use a video file or 'video_capture = cv2.VideoCapture(0)' use camera
 
 speed : when only run yolo detection about 11-13 fps  , after add deep_sort about 11.5 fps
 
 test video : https://www.bilibili.com/video/av23500163/
 
 From the issue https://github.com/Qidian213/deep_sort_yolov3/issues/7 , it can tracks cars, birds and trucks too and performs well .



