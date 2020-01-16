'''This is the main entry point for the pallet loading solution.

For examples on how to execute this script refer to CLI_Args.txt

'''

from utils.app_utils import FPS, WebcamVideoStream, HLSVideoStream
from multiprocessing import Queue, Pool, Value, Lock
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.trackAndDetect import trackingPoolWorker as tpw
from object_detection.ocr import readers
from object_detection.ocr.thumbnail import Thumbnails
import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-fName',"--folderName", type=str,
                       help="name of directory to store thumbnails", required=True)
    parser.add_argument('-str', '--stream', dest="stream", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int, default=0, help='Device index of the camera.')
    parser.add_argument("-v", "--video", help="path to input video file")
    parser.add_argument("-o", "--output", type=str,
	                    help="path to store solution results")
    parser.add_argument("-to", "--tempStorage", required=True, help="path to save interim tracker output")
    parser.add_argument("-c", "--confidence", type=float, help="filter for low detection scores", default=0.9)
    parser.add_argument("-sf", "--skipFrames", type=int, help="apply detection on every nth rame", default=20)
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    parser.add_argument('-dbg', '--debug', type=bool, help="display additional output to help debug code", default=False)
    parser.add_argument('-bf', '--buffer', type=int, default=0,
                        help="buffer to be added to coordinate threshold for distinguishing between objects")
    parser.add_argument('-cl', '--counterline', type=float,
                        help="percentage of frame an object must traverse before the counter is incremented")
    parser.add_argument('-dr', '--direction', help="direction of movement, valid options are t2b, b2t, l2r and r2l", default='t2b')
    parser.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
    parser.add_argument("-ctxt", "--min-confidence", type=float, default=0.5,
	                    help="minimum probability required to inspect a region for text")
    parser.add_argument("-ocrW", "--ocrWidth", type=int, default=320,
	                    help="nearest multiple of 32 for resized width")
    parser.add_argument("-ocrH", "--ocrHeight", type=int, default=320,
	                    help="nearest multiple of 32 for resized height")
    parser.add_argument("-p", "--padding", type=float, default=0.0,
	                    help="amount of padding to add to each border of ROI")
    parser.add_argument("-ct", "--codeType", type=str,
                    default="barcode",help="options are ocr, barcode or qrcode")

    args = parser.parse_args()

    debug_status = "True" if vars(args).get("debug") == True else "False"
    conveyorDirection = vars(args).get("direction")

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
    #mapping of label index to human readable class
    category_index = label_map_util.create_category_index(categories)

    #global variable for assigning tracker IDs
    nextTrackerID = tpw.CounterVar()

    #object counter variable
    numObjDetected = tpw.CounterVar()

    #global variable for assigning image IDs
    nextImageID = tpw.CounterVar()

    #create tensorflow session and model graph
    sess, detection_graph = tpw.create_tf_sess(PATH_TO_CKPT)

    #global variable to track bounding box threshold
    if args.direction == 't2b' or args.direction == 'l2r':
        coordThreshold = tpw.BBoxCoordinate(initval=0)
    elif args.direction == 'b2t':
        coordThreshold == tpw.BBoxCoordinate(initval=args.height)
    elif args.direction == 'r2l':
        coordThreshold = tpw.BBoxCoordinate(initval=args.width)

    #create results folder structure
    result_root = os.path.join(args.output,args.folderName)
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    pool = Pool(args.num_workers, tpw.dtc_worker, (args.folderName, result_root, PATH_TO_CKPT, input_q, output_q, coordThreshold,category_index,args.skipFrames,args.confidence, args.tempStorage, nextTrackerID, args.buffer, numObjDetected, args.counterline, args.direction, args.height, args.width, nextImageID))

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

    writer = None
    t = time.time()

    frameNum = 0

    # start the frames per second throughput estimator
    fps = FPS().start()

    while True:  # fps._numFrames < 120
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

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args.output is not None and writer is None:

            video_path = os.path.join(result_root, "Processed_Video")
            #check if the directory exists
            if not os.path.exists(video_path):
                os.makedirs(video_path)

            video_file_name = os.path.join(video_path,(args.folderName + ".avi"))

            fourcc = cv2.cv2.VideoWriter_fourcc(*"MJPG") #VideoWriter_fourcc(*'MP4V')
            writer = cv2.VideoWriter(video_file_name, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        #output_rgb = cv2.resize(output_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        if debug_status == "True":
            #the line after which an object detected will be considered a new instance
            tpw.drawReferenceLine(output_rgb,coordThreshold, args.direction, buffer=0)
            if args.buffer > 0:
                #buffer line
                tpw.drawReferenceLine(output_rgb,coordThreshold, args.direction, buffer=args.buffer)

        #counter threshold line
        tpw.drawCounterLine(output_rgb, args.counterline, args.direction)

        #show object count
        text = f'Box count: {numObjDetected.value}'
        cv2.putText(output_rgb, text, (10, int(output_rgb.shape[0] * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow('Video', output_rgb)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(output_rgb)

        frameNum += 1

        fps.update()

        #if debug_status == "True":
        #    print(f'[INFO] Current box count: {numObjDetected.value}')
        #    print(f'[INFO] Current coordinate threshold value : {coordThreshold.value}')
        
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    fps.stop()
    
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    
    log_path = os.path.join(result_root, "Logs")
    #check if the directory exists
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = os.path.join(log_path,(args.folderName + ".log"))

    with open(log_file_name, 'a+') as f:
        print(f'[INFO] conveyor direction: {conveyorDirection}', file=f)
        print(f'[INFO] debug mode: {debug_status}',file=f)
        print(f'[INFO] elapsed time (total): {fps.elapsed():.2f} seconds',file=f)
        print(f'[INFO] approx. FPS: {fps.fps():.2f}',file=f)
        print(f'[INFO] total box count: {numObjDetected.value}',file=f)

    pool.terminate()

    if (args.video):
        video_capture.release()
    else:
        video_capture.stop()

    cv2.destroyAllWindows()

    #this block of code will read CBU code, barcode or QR code depending on the input parameter
    start = time.time()

    thumbnailDir = os.path.join(result_root, "Thumbnails")
    #check if the directory exists
    if not os.path.exists(thumbnailDir):
        os.makedirs(thumbnailDir)

    ocrSaveDir = os.path.join(result_root, "Product_Code_Detection")
    if not os.path.exists(ocrSaveDir):
        os.mkdir(ocrSaveDir)

    #retrieve list of image files for which we will read the product codes
    image_files_list = [f for f in os.listdir(thumbnailDir) if os.path.isfile(os.path.join(thumbnailDir, f))]

    thumbnails = Thumbnails(imgSourcePath=thumbnailDir, thumbnailList=image_files_list, saveDir=ocrSaveDir, resize_height=args.ocrHeight, resize_width=args.ocrWidth, tf_model_path=args.east, ocr_confidence=args.min_confidence, padding=args.padding)

    pCode_reader = readers.ObjectReader()

    pCode_reader.read(readable=thumbnails, format=args.codeType)

    with open(log_file_name, 'a+') as f:
        print(f'[INFO] product code type: {args.codeType}', file=f)
        print(f'[INFO] total time taken for reading product codes: {time.time() - start:.2f} seconds', file=f)

    print(f'Processing completed!')