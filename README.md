
# Introduction
Thanks for these projects, this work now is support tiny_yolo v3 but only for test, if you want to train you can either train a model in darknet or in the second following works. It also can tracks many objects in coco classes, so please note to modify the classes in yolo.py. besides, you also can use camera for testing.

  https://github.com/nwojke/deep_sort
  
  https://github.com/qqwweee/keras-yolo3
  
  https://github.com/Qidian213/deep_sort_yolov3

# Quick Start

1. Download YOLOv3 or tiny_yolov3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO_DEEP_SORT 

The following three steps, you can change accordingly:

```
   please download the weights at first from yolo website or use your own weights. 
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
 
# Test only

 speed : when only run yolo detection about 11-13 fps  , after add deep_sort about 11.5 fps
 
 test video : https://www.bilibili.com/video/av23500163/
 

