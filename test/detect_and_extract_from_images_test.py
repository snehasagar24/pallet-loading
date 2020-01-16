import os
import cv2
import time
import argparse
import logging
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.trackAndDetect import imageExtraction as ime
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str,
	                    help="path to input image file")
    parser.add_argument("-o", "--output", type=str,
	                    help="path to optional output video file")
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.') 
    parser.add_argument("-c", "--confidence", type=float, help="filter for low detection scores", default=0.9)
    args = parser.parse_args()

    # Read and preprocess an image.
    print("[INFO] reading image...")
    img = cv2.imread(args.image)
    img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)

    # convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #perform object detection
    output_rgb, trackableObj = tpw.detect_objects(PATH_TO_CKPT, img_rgb, category_index, args.confidence)

    #enumerate over the scores and filter out weak detections
    high_confidence_indexes = [i for i in range(trackableObj._scores.shape[1]) if trackableObj._scores[0][i] >= args.confidence]

    #proceed only if there are detections greater than the threshold
    if high_confidence_indexes:
        #extract thumbnail
        for idx in high_confidence_indexes:
            each_box = trackableObj._boxes[0][idx]
            normalized_bbox = tpw.normalizeBBoxCoordinates(each_box,args.height,args.width,debug=False)
            cropped_image = ime.getThumbnail(img, normalized_bbox)

            cv2.imwrite(args.output, cropped_image)
            cv2.imshow('Cropped Image', cropped_image)
            cv2.waitKey()

    cv2.destroyAllWindows()
    print("[INFO] object detection completed!")