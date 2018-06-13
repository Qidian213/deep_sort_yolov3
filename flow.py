from tools import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from ReadImage import ReadImageThread
from DetectImage import DetectImageThread
from yolo import YOLO
import queue
from Output import OutputThread
from video import VideoReader
from keras import backend as K
from timeit import time as timeit
import time
from tools.QsizeDebugger import QsizeDebugger

def main():
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = 5
    nms_max_overlap = 1.0

    # deep_sort 
    model_filename = 'model_data/mars-small128.pb'

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False

    display_flag = True

    videoReader = VideoReader("./test0604s.mp4")

    const_frame = videoReader.const_frame

    # 帧间隔时间 毫秒
    frame_interval = 1000 / videoReader.fps

    # 轨迹图
    trackmap = None
    trackstorage = {}

    fps = 0.0
    finash_frame = 0

    sess = K.get_session()
    yolo = YOLO(sess)
    encoder = gdet.create_box_encoder(model_filename,batch_size=32, session=sess)

    readImage = ReadImageThread(1, videoReader)

    outputQueue = queue.Queue(maxsize=110)

    detectImageWorkers = [DetectImageThread(yolo, encoder, readImage.queue, outputQueue) for x in range(100)]

    print('Start %d detect image workers' % len(detectImageWorkers))

    outputWorker1 = OutputThread(outputQueue, tracker, nms_max_overlap=nms_max_overlap)

    start = timeit.time()

    readImage.start()
    [worker.start() for worker in detectImageWorkers]
    outputWorker1.start()

    inputDebugger = QsizeDebugger('input', readImage.queue)
    inputDebugger.start()

    outputDebugger = QsizeDebugger('output', outputQueue)
    outputDebugger.start()

    readImage.join()
    print('视频读取完成')
    while not readImage.queue.empty():
        time.sleep(0.01)
        continue
    [worker.stop() for worker in detectImageWorkers]
    [worker.join() for worker in detectImageWorkers]
    print('视频处理完成')
    while not outputQueue.empty():
        time.sleep(0.01)
        continue
    outputWorker1.stop()
    outputWorker1.join()
    print('视频输出完成')

    inputDebugger.stop()
    outputDebugger.stop()

    take_time = timeit.time() - start
    print('takeTime = %02d:%02d:%02d' % (take_time / 3600, take_time / 60 % 60, take_time % 60))
    print('fps = %f' % (1. / (take_time / 1200.)))

if __name__ == '__main__':
    main()
