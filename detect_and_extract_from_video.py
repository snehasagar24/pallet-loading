import os
import cv2
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool, Value, Lock
from utils.app_utils import FPS, WebcamVideoStream, HLSVideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.trackAndDetect import trackingPoolWorker as tpw

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'conveyorBelt'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'ul_label_map.pbtxt')

NUM_CLASSES = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#global variable for assigning image IDs
nextImageID = tpw.CounterVar()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-str', '--stream', dest="stream", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int, default=0, help='Device index of the camera.')
    parser.add_argument("-v", "--video", help="path to input video file")
    parser.add_argument("-o", "--output", type=str,
	                    help="path to optional output object images")
    parser.add_argument("-c", "--confidence", type=float, help="filter for low detection scores", default=0.9)
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    parser.add_argument('-lf', '--log', help="path to log file", required=True)
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    pool = Pool(args.num_workers, tpw.thumbnail_worker, (args.video, input_q, output_q, PATH_TO_CKPT, category_index, args.confidence, args.height, args.width, nextImageID, args.output))

    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")

    if (args.video):
        print('Reading from video file.')
        video_capture = cv2.VideoCapture(args.video)
    
    elif (args.stream):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream).start()

    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()

    t = time.time()

    frameNum = 0

    # start the frames per second throughput estimator
    fps = FPS().start()

    while True:
        if (args.video):
            (grabbed, frame) = video_capture.read()
            #resize frames to speed up processing
            if frame is not None:
                (H,W) = frame.shape[:2] #get original height and width of frame
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        else:
            frame = video_capture.read()        

        input_q.put(tpw.InputQueueObject(frame,frameNum))

        # check to see if we have reached the end of the video file
        if frame is None:
            break

        #cv2.imshow("Video Frame",frame)

        frameNum += 1

        fps.update()

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    fps.stop()
    
    with open(args.log, 'w') as f:
        print(f'[INFO] elapsed time (total): {fps.elapsed():.2f}',file=f)
        print(f'[INFO] approx. FPS: {fps.fps():.2f}',file=f)

    pool.terminate()

    if (args.video):
        video_capture.release()
    else:
        video_capture.stop()

    cv2.destroyAllWindows()