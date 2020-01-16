# import the necessary packages
from multiprocessing import Queue, Process, Pool
from object_detection.utils import label_map_util
from object_detection.trackAndDetect.zoomFrame import Zoom
from object_detection.trackAndDetect import trackerZoo as tkz
from object_detection.utils import visualization_utils as vis_util
from utils.app_utils import FPS, WebcamVideoStream, HLSVideoStream
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to TensorFlow pre-trained model")
ap.add_argument("-L", "--label", required=True,
                help='name of label pbtxt file')
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
ap.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
ap.add_argument("-f", "--fline", type=float, default=0.6,
                help="region in the video frame after which we count an object")
ap.add_argument("-w", "--width", type=int, default=500,
                help="width of the resized video frame")
ap.add_argument("-l", "--log", type=str, required=True,
                help="file to write results of processing")

args = vars(ap.parse_args())

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#MODEL_NAME = 'conveyorBelt'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', args["model"], 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', args["label"])

NUM_CLASSES = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def main():

    # initialize our list of queues -- both input queue and output queue
    # for *every* object that we will be tracking
    inputQueues = []
    outputQueues = []

    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(args["input"])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # start the frames per second throughput estimator
    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else Zoom(frame, 15)

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        #convert the frame from BGR to RGB for dlib and passing to the TF model
        frame = cv2.resize(frame, (args["width"],args["height"]), cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        # if our list of queues is empty then we know we have yet to create our
        # first object tracker
        if len(inputQueues) == 0:

            image_np, boxes, scores, classes, num_detections = tkz.box_detector(PATH_TO_CKPT, category_index, rgb)

            #enumerate over the scores and filter out weak detections
            high_confidence_indexes = [i for i in range(scores.shape[1]) if scores[0][i] >= args["confidence"]]

            for idx in high_confidence_indexes:
                category_idx = classes[0][idx]
                label = category_index[category_idx]["name"]
                box = boxes[0][idx]
                bb = (startX, startY, endX, endY) = box.astype("int")
                score = scores[0][idx]

                # create two brand new input and output tracking queues,
                # respectively
                iq = Queue()
                oq = Queue()
                inputQueues.append(iq)
                outputQueues.append(oq)

                # spawn a daemon process for a new object tracker
                p = Process(target=tkz.start_tracking,
                args=(PATH_TO_CKPT, category_index, bb, rgb, iq, oq))
                p.daemon = True
                p.start()
                
        # otherwise, we've already performed detection so let's track
        # multiple objects
        else:
            # loop over each of our input ques and add the input RGB
            # frame to it, enabling us to update each of the respective
            # object trackers running in separate processes
            for iq in inputQueues:
                iq.put(rgb)

            # loop over each of the output queues
            for oq in outputQueues:
                # grab the updated bounding box coordinates for the
                # object -- the .get method is a blocking operation so
                # this will pause our execution until the respective
                # process finishes the tracking update
                image_np = oq.get()
                text = f'Count: {len(outputQueues)}'
                cv2.putText(image_np, text, (int(args["width"]/2), args["height"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # show the output frame
        cv2.imshow("Frame", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] Total boxes: {} ".format(len(outputQueues)))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()
    p.terminate()

if __name__ == '__main__':
    main()
