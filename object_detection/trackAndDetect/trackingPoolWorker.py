from object_detection.utils import visualization_utils as vis_util
from utils.app_utils import FPS
from multiprocessing import Queue
from os import listdir
from os.path import isfile, join
from object_detection.trackAndDetect import imageExtraction as ime
import os
import pickle
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib
import cv2
import tensorflow as tf

class InputQueueObject():
    '''Custom class for objects held in the input queue

    Parameters:

        rgb: array<float> - image/video frame

        frameNum: int - corresponding video frame number. This will be useful for doing Skip-Frame Detections'''

    def __init__(self, rgb, frameNum):
        self._rgb = rgb
        self._frameNum = frameNum

class ThumbNailObject(InputQueueObject):
    '''Custom class containing a cropped image and it's meta data'''

    def __init__(self, fileName, frameNum, imgNum, rgb):
        self._fileName = fileName
        self._imgNum = imgNum

        InputQueueObject.__init__(self, rgb, frameNum)

class TrackableObject():
    '''Custom class to hold the bounding box coordinates, detection score, label class index and label mapping dictionary
    for each object detected in an image or frame
    
    Parameters:

        boxes: array<int> - multi-dimensional array of shape (1,n,4) representing the coordinates for the bounding box of an object

        scores: array<float> - prediction score assigned by the object detection model. It is an array of shape (1,n)

        classes: array<float> - an array of shape (1,n) containing the label index for each detected object

        category_index: dict<int,str> - a dictionary that maps the label index to the class or human readable label
        '''
    def __init__(self, boxes, scores, classes):
        self._boxes = boxes
        self._scores = scores
        self._classes = classes

class TrackerQueueObject():
    '''Custom class to hold an instance of TrackableObject and it's corresponding rectangle coordinates from a tracker and the corresponding image. This class will be processed in a multiprocess tracking queue'''

    def __init__(self, id, trackableObject, tracker_rect, rgb):
        self._id = id
        self._trackableObject = trackableObject
        self._tracker = tracker_rect
        self._original_image = rgb

class BBoxCoordinate(object):
    '''This class will keep a track of the maximum x (width) or y (height) coordinate amongst a group of bounding box coordinates tracked in an image/video frame. When detection is performed on frames following the first, then only objects with a starting x or y coordinate less/ greater than the coordinate will be considered new objects and new trackers will be created for them'''

    def __init__(self, initval=0):
        self.initval = initval
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()
        #self.numResets = multiprocessing.RawValue('i', initval)

    def update(self, coord):
        with self.lock:
            self.val.value = coord

    def reset(self):
        with self.lock:
            self.val.value = self.initval

    @property
    def value(self):
        return self.val.value

class CounterVar(object):
    '''This class can be used for counter variables that need to be shared across processes'''

    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def update(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value

def create_tf_sess(PATH_TO_CKPT):
    '''Load a (frozen) Tensorflow model into memory.

    Parameters:

        PATH_TO_CKPT: str - location of directory where the trained model is stored on disk in Protobuff (.pb) format

    Returns:

        sess: TensorFlow Session instance

        detection_graph: In memory TensorFlow Graph object of the loaded model'''

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    return sess, detection_graph

def detect_objects(modelPath, image_np, category_index,confidence):
    '''This method applies the object detection model on an image/video frame and detects all
    instances of the classes it has been trained on. The output from all the detections are then passed along to
    another helper function which annotates the image/video frame with bouding boxes, class name and detection score.

    Parameters:

        image_np: array<float> - the image or video frame on which object detection needs to be applied

        category_index: dict<int,str> - a dictionary that maps the label index to the class or human readable label

        modelPath: str - location where the TensorFlow model is saved

    Returns:

        image_np: annotated image

        boxes: array<float> - multi-dimensional array of shape (1,n,4) representing the coordinates for the bounding box of an object

        scores: array<float> - prediction score assigned by the object detection model. It is an array of shape (1,n)

        classes: array<float> - an array of shape (1,n) containing the label index for each detected object'''

    sess, detection_graph = create_tf_sess(modelPath)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=confidence)

    sess.close()

    return image_np, TrackableObject(boxes, scores, classes)

def start_trackers(trackerIdObj, rgb, trackableObj, category_index, confidence, tempOutPath, image_height, image_width, fileName, frameNum, nextImageID, thumbnailPath):
    '''This method will take the detection results from an image/video frame and create individual instances of
    correlational trackers if the detection score exceeds a pre-determined threshold. Once created the trackers together
    with their corresponding bounding box coordinates, score, class and category_index are wrapped in a TrackableQueueObject class instance and added to the tracking queue

    Parameters:

        trackerIdObj: int - global variable which holds the next available unique ID that can be assigned a new tracker

        rgb: array<float> - image/video frame

        trackableObj: an instance of the TrackableObject class

        category_index: dict<int,str> - a dictionary that maps the label index to the class or human readable label

        confidence: array<float> - float, threshold value for object detection

        tempOutPath: str - temporary location on disk to store TrackingQueueObjects

        image_height: int - height of the image/video frame

        image_width: int - width of the image/video frame
        '''
    
    #enumerate over the scores and filter out weak detections
    high_confidence_indexes = [i for i in range(trackableObj._scores.shape[1]) if trackableObj._scores[0][i] >= confidence]
    #print(f'Indexes of high detections: {high_confidence_indexes}')

    #proceed only if there are detections greater than the threshold
    if high_confidence_indexes:
    #if np.array(high_confidence_indexes).size != 0:

        for idx in high_confidence_indexes:
            category_idx = trackableObj._classes[0][idx]
            label = category_index[category_idx]["name"]
            box = trackableObj._boxes[0][idx]
            box = normalizeBBoxCoordinates(box,image_height,image_width)
            score = trackableObj._scores[0][idx]

            # construct a dlib rectangle object from the bounding box
	        # coordinates and then start the correlation tracker
            trckr = dlib.correlation_tracker()

            '''NOTE:
                ymin = top
                xmin = left
                ymax = bottom
                xmax = right
                TensorFlow requires bounding boxes in the format [ymin, xmin, ymax, xmax]
                dlib.rectangle requires boxes in the format [left,top,right,bottom]'''

            rect = dlib.rectangle(box[1], box[0], box[3], box[2])
            trckr.start_track(rgb, rect)
            trackableObj = TrackableObject(boxes=np.array([[box]]), scores = np.array([[score]]),
                                 classes= np.array([[category_idx]]))
            trackerQueueObj = TrackerQueueObject(trackerIdObj.value, trackableObj,trckr.get_position(),rgb)
            trackerIdObj.update()
            #Write each updated tracker object to the interim output directory
            with open(join(tempOutPath,("trackerID" + str(trackerQueueObj._id) + ".pkl")), "wb") as fout:
                pickle.dump(trackerQueueObj,fout, protocol=pickle.HIGHEST_PROTOCOL)

            #extract and write object thumbnail to disk
            cropped_image = ime.getThumbnail(rgb,box)
            ime.writeThumbnailToDisk([cropped_image], fileName, frameNum, nextImageID, thumbnailPath)

        return True

    else:
        return False

def update_tracker(rgb, pickledTrackers, tempOutPath, category_index, coord_threshold, output_q, detectedObjCounter, counterline, direction, purpose, confidence):
    '''This method will update the position of a previously detected box

    Parameters:

        rgb: array<float> - image/video frame

        pickledTrackers: list<str> - list of serialized TrackingQueueObjects

        tempOutPath: str - temporary location on disk to store TrackingQueueObjects

        category_index: dict<int,str> - a dictionary that maps the label index to the class or human readable label

        coordthreshold: int - coordinate below or above which qualifies for creating new object trackers

        output_q: an instance of a multi process queue containing multiple instances of TrackableObjects. The latter will be used to annotate the image/frame that will utlimately be written to disk

        detectedObjCounter: int - global counter variable to keep track of objects counted in the video

        counterline: float - region in the image, after which the object counter is incremented for an objct crossing it

        direction: str - which direction is the conveyor moving in relation to the video, valid options are -

            't2b' - top to bottom

            'b2t' - bottom to top

            'l2r' - left to right

            'r2l' - right to left

    Returns:

        rgb: annotated image'''

    boxes = []
    scores = []
    classes = []

    for each_file in pickledTrackers:

        fileName = join(tempOutPath, each_file)

        with open(fileName, "rb") as file:
            trackerQueueObj = pickle.load(file)
  
        #instantiate a tracker
        tracker = dlib.correlation_tracker()
        tracker.start_track(trackerQueueObj._original_image, trackerQueueObj._tracker)
        #update the bounding box coordinates
        tracker.update(rgb)        

        #get the rectangle object from the updated tracker
        pos = tracker.get_position()

        #update the rectangle coordinates and the new image
        trackerQueueObj._tracker = pos
        #update the original image attribute with the latest frame
        trackerQueueObj._original_image = rgb

        # unpack the position object
        left = int(pos.left())
        top = int(pos.top())
        right = int(pos.right())
        bottom = int(pos.bottom())
        #print(f'Updated coordinates - left: {left}, top: {top}, right: {right}, bottom: {bottom}')

        #top to bottom conveyor belt movement
        if direction == 't2b' and top > coord_threshold.value:
            coord_threshold.update(top)
        #bottom to top conveyor belt movement
        elif direction == 'b2t' and bottom < coord_threshold.value:
            coord_threshold.update(bottom)
        #right to left conveyor belt movement
        elif direction == 'r2l' and right < coord_threshold.value:
            coord_threshold.update(right)
        #left to right conveyor movement
        elif direction == 'l2r' and left > coord_threshold.value:
            coord_threshold.update(left)

        #check if counter needs to be incremented
        if direction == 't2b' or direction == 'b2t':
            coord1 = top
            coord2 = bottom
        else:
            coord1 = left
            coord2 = right

        if crossedCounterLine(rgb,coord1,coord2,counterline,direction,purpose):
            detectedObjCounter.update()
            os.remove(fileName)
            #coord_threshold.reset(direction,rgb)

            #reset the coordinate threshold if the counted object's relevant coordinate is equal to the coordinate threshold
            if direction == 't2b' and top == coord_threshold.value:
                coord_threshold.reset()
            #bottom to top conveyor belt movement
            elif direction == 'b2t' and bottom == coord_threshold.value:
                coord_threshold.reset()
            #right to left conveyor belt movement
            elif direction == 'r2l' and right == coord_threshold.value:
                coord_threshold.reset()
            #left to right conveyor movement
            elif direction == 'l2r' and left == coord_threshold.value:
                coord_threshold.reset()
            
        else:
            #Write each updated tracker object to the interim output directory
            with open(fileName, "wb") as fout:
                pickle.dump(trackerQueueObj,fout,protocol=pickle.HIGHEST_PROTOCOL)

            '''NOTE:
                ymin = top
                xmin = left
                ymax = bottom
                xmax = right
                TensorFlow requires bounding boxes in the format [ymin, xmin, ymax, xmax]
                dlib.rectangle requires boxes in the format [left,top,right,bottom]'''

            #update the lists
            boxes.append([top, left, bottom, right])
            scores.append(trackerQueueObj._trackableObject._scores[0][0])
            classes.append(trackerQueueObj._trackableObject._classes[0][0])

            '''update the global variable for the coordinate threshold
            Key Point - y-axis values increase from top to bottom of the image while x-axis values increase from left to right of the image'''

    #annotate the image
    vis_util.visualize_boxes_and_labels_on_image_array(
    rgb,
    np.array(boxes),
    np.array(classes).astype(np.int32),
    np.array(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=confidence)

    output_q.put(rgb) #annotated image

def normalizeBBoxCoordinates(bbox,image_height,image_width,debug=False):
    '''normalizing the coordinates with respect to the original frame size. Since Tensorflow returns bounding box in the format [ymin, xmin, ymax, xmax], hence the normalizer will be an array of format [numRows,numCols,numRows,numCols]
    
    Parameters:
    
        bbox: array<int>  - coordinates of the 4 end points of the rectangle that bounds a detected object
        
        image_height: int - the height of the original image passed to the object detection model
        
        image_width: int  - the width of the original image passed to the object detection model
        
    Returns
    
        box: array<int> - scaled coordinates of the 4 end points of the rectangle that will bound the object in the original image
        '''

    box = bbox * np.array([image_height, image_width, image_height, image_width])
    if debug:
        print(box)
    box = box.astype("int").tolist()
    if debug:
        print(box)

    return box

def getDuplicateObjCoord(rgb,trackableObj,coordThreshold,image_height,image_width,buffer,counterline,direction,purpose):
    '''This function will compare the threshold value with the corresponding box coordinate to determine if the object detected is new or an existing one currently being tracked'''

    delete_index = []

    for idx, each_box in enumerate(trackableObj._boxes[0]):

        each_box = normalizeBBoxCoordinates(each_box,image_height,image_width,debug=False)

        '''assuming that boxes move from top to bottom of the screen the max_y (bottom) coordinate of each box should be less than the threshold value (plus some buffer, since trackers are predicting the next position of the tracked object) to be considered a new object for which a new tracker should be started the top left corner of an image is (0,0) and the bottom is (maxX,maxY)'''
                        
        #print(f'Rectangle coordinates: {each_box[3]}')
        #print(f'Buffer line value: {coordThreshold.value - 100}')
        #print(f'New object detection threshold value: {coordThreshold.value}')

        '''NOTE:
                ymin = top =    box[0]
                xmin = left =   box[1]
                ymax = bottom = box[2]
                xmax = right =  box[3]

                TensorFlow requires bounding boxes in the format [ymin, xmin, ymax, xmax]
                dlib.rectangle requires boxes in the format [left,top,right,bottom]'''

        if direction == 't2b' or direction == 'b2t':
            coord1 = each_box[0]
            coord2 = each_box[2]
        else:
            coord1 = each_box[1]
            coord2 = each_box[3]

        #check if the new objects detected are beyond or before the threshold depnding on direction
        if direction == 't2b' and each_box[2] > (coordThreshold.value - buffer):
            delete_index.append(idx)
        elif direction == 'b2t' and each_box[0] < (coordThreshold.value + buffer):
            delete_index.append(idx)
        elif direction == 'l2r' and each_box[3] < (coordThreshold.value - buffer):
            delete_index.append(idx)
        elif direction == 'r2l' and each_box[1] > (coordThreshold.value + buffer):
            delete_index.append(idx)
        #also check if the new box is detected after the counter line - region after which counter is incremented
        elif crossedCounterLine(rgb,coord1,coord2,counterline,direction,purpose):
            delete_index.append(idx)
    
    #print(f'Indexes of duplicate boxes to be deleted: {np.array(delete_index)}')
    #print()

    return np.array(delete_index)

def crossedCounterLine(rgb, coord1, coord2, counterline, direction, purpose='counting'):
    '''Method to determine if the object counter should be increased depending on the position of an object in relation to the count region threshold

    Parameters:

        rgb: array<float> - image/video frame

        coord1: int - minY/minX coordinate of a tracked object

        coord2: int - maxY/maxX coordinate of a tracked object

        counterline: float - value that represents the region on the image/video frame after which we want to count an object after detecting and tracking it across multiple frames

        direction: str - which direction is the conveyor moving in relation to the video, valid options are -

            't2b' - top to bottom

            'b2t' - bottom to top

            'l2r' - left to right

            'r2l' - right to left
            '''

    if direction == 't2b' or direction == 'b2t':
        cutoff = int(rgb.shape[0] * counterline)
    else:
        cutoff = int(rgb.shape[1] * counterline)

    if purpose == 'counting':

        mid = int((coord1+coord2)/2)

        if direction == 't2b' or direction == 'l2r':
            return mid >= cutoff
        elif direction == 'b2t' or direction == 'r2l':
            return mid < cutoff
    
    elif purpose == 'dedup':

        if direction == 't2b'or direction == 'l2r':
            return coord2 > cutoff #check where the bottom/right edge of the object is in relation to the counter line
        elif direction == 'b2t' or direction == 'r2l':
            return coord1 < cutoff

def drawReferenceLine(rgb,coordThreshold, direction, buffer=0):
    '''This method renders the boundary line which separates newly detected objects from ones which were detected in an earlier frame

    Parameters:

    rgb: array<float> - image/video frame

    coordThreshold: int - this is the line which distinguishes between newly detected objects from existing ones

    direction: str - which direction is the conveyor moving in relation to the video, valid options are -

            't2b' - top to bottom

            'b2t' - bottom to top

            'l2r' - left to right

            'r2l' - right to left 
    '''

    #horizontal line
    if direction == 't2b':       
        cv2.line(rgb, (0, coordThreshold.value - buffer), (rgb.shape[1], coordThreshold.value - buffer), (0, 255, 255), 1)
    elif direction == 'b2t':        
        cv2.line(rgb, (0, coordThreshold.value + buffer), (rgb.shape[1], coordThreshold.value + buffer), (0, 255, 255), 1)

    #vertical line
    elif direction == 'r2l':        
        cv2.line(rgb, (coordThreshold.value + buffer, 0), (coordThreshold.value + buffer, rgb.shape[0]), (0, 255, 255), 1)
    elif direction == 'l2r':        
        cv2.line(rgb, (coordThreshold.value - buffer, 0), (coordThreshold.value - buffer, rgb.shape[0]), (0, 255, 255), 1)

def drawCounterLine(rgb,counterline,direction):
    '''This method renders the line, which when crossed increments the object counter
    
    Parameters:

    rgb: array<float> - image/video frame

    counterline: float - value that represents the region on the image/video frame after which we want to count an object after detecting and tracking it across multiple frames

    direction: str - which direction is the conveyor moving in relation to the video, valid options are -

            't2b' - top to bottom

            'b2t' - bottom to top

            'l2r' - left to right

            'r2l' - right to left '''

    #horizontal line
    if direction == 't2b' or direction == 'b2t':
        cv2.line(rgb, (0, int(rgb.shape[0] * counterline)), (rgb.shape[1], int(rgb.shape[0] * counterline)), (255, 0, 0), 1)

    #vertical line
    else:
        cv2.line(rgb, (int(rgb.shape[1] * counterline), 0), (int(rgb.shape[1] * counterline), rgb.shape[0]), (255, 0, 0), 1)
    
def dtc_worker(inputFilePath, result_root, modelPath, input_q, output_q, coordThreshold, category_index, skip_frames,
           confidence, tempOutPath, trackerObjID, coordBuffer, detectedObjCounter, counterline, direction, image_height, image_width, nextImageID):
    '''This is the pool worker which will orchestrate the entire operations. dtc - detect, track, count'''

    fps = FPS().start()

    while True:
        fps.update()
        inputQueueObj = input_q.get()
        if inputQueueObj._rgb is not None:
            frame_rgb = cv2.cvtColor(inputQueueObj._rgb, cv2.COLOR_BGR2RGB)

            (height_y, width_x) = frame_rgb.shape[:2]

            #run detection on the first frame
            if inputQueueObj._frameNum == 0:
                image_np, trackableObj = detect_objects(modelPath,frame_rgb,category_index,confidence)
                output_q.put(image_np)
                start_trackers(trackerObjID,frame_rgb,trackableObj,category_index,confidence,tempOutPath, height_y, width_x, inputFilePath, inputQueueObj._frameNum, nextImageID, result_root)

            #run subsequent detections if frame number is a multiple of the skip frames parameter
            elif inputQueueObj._frameNum % skip_frames == 0:

                image_np, trackableObj = detect_objects(modelPath,frame_rgb,category_index,confidence)
                output_q.put(image_np)

                #for subsequent detections we have to avoid double counting
                #check if the treshold value is not at it's initialization value
                if coordThreshold.value != coordThreshold.initval:
                    #get numpy array of duplicate object indexes
                    delete_index = getDuplicateObjCoord(frame_rgb,trackableObj,coordThreshold,height_y,width_x,coordBuffer,counterline,direction,"dedup")

                    if delete_index.size != 0:
                        #delete all trackable object instances that do not exceed the threshold
                        trackableObj._boxes = np.delete(trackableObj._boxes, delete_index, axis=1)
                        trackableObj._scores = np.delete(trackableObj._scores, delete_index, axis=1)
                        trackableObj._classes= np.delete(trackableObj._classes, delete_index, axis=1)

                        #check if the trackableObj is empty
                        if trackableObj._boxes.size != 0:

                            #start separate trackers for each valid new object detected
                            reset_threshold = start_trackers(trackerObjID,frame_rgb,trackableObj,category_index,confidence,tempOutPath, height_y, width_x, inputFilePath, inputQueueObj._frameNum, nextImageID, result_root)

                            if reset_threshold:
                                coordThreshold.reset()

                    else:
                        reset_treshold = start_trackers(trackerObjID,frame_rgb,trackableObj,category_index,confidence,tempOutPath, height_y, width_x, inputFilePath, inputQueueObj._frameNum, nextImageID, result_root)

                        if reset_threshold:
                                coordThreshold.reset()

                else:
                    reset_threshold = start_trackers(trackerObjID,frame_rgb,trackableObj,category_index,confidence,tempOutPath, height_y, width_x, inputFilePath, inputQueueObj._frameNum, nextImageID, result_root)

                    if reset_threshold:
                                coordThreshold.reset()

            #run tracking if the frame number is not a multiple of the skip frames parameter
            else:

                #if there aren't any objects being tracked yet then return the image/video frame
                if not os.listdir(tempOutPath):
                    output_q.put(frame_rgb)
                else:
                    onlyfiles = [f for f in listdir(tempOutPath) if isfile(join(tempOutPath, f))]
                    update_tracker(frame_rgb,onlyfiles,tempOutPath,category_index, coordThreshold, output_q, detectedObjCounter, counterline,direction,"counting",confidence)

    fps.stop()

def thumbnail_worker(inputFilePath, input_q, output_q, modelPath, category_index, confidence, image_height, image_width, imgNum, saveDir):
    '''This pool worker will coordinate all tasks from getting an image array from a queue, extracting an object thumbnail and writing the same to a location on disk'''

    fps = FPS().start()

    while True:
        fps.update()
        inputQueueObj = input_q.get()
        
        if inputQueueObj._rgb is not None:

            try:

                if inputQueueObj._frameNum % 10 == 0:

                    # convert to RGB
                    img_rgb = cv2.cvtColor(inputQueueObj._rgb, cv2.COLOR_BGR2RGB)
                    #retrieve file name from the file path
                    fileName = os.path.basename(inputFilePath).split('.')[0]

                    #perform object detection
                    output_rgb, trackableObj = detect_objects(modelPath, img_rgb, category_index, confidence)

                    #enumerate over the scores and filter out weak detections
                    high_confidence_indexes = [i for i in range(trackableObj._scores.shape[1]) if trackableObj._scores[0][i] >= confidence]

                    #proceed only if there are detections greater than the threshold
                    if high_confidence_indexes:

                        #extract thumbnail
                        thumbnail_list = ime.thumbnailIterator(img_rgb, high_confidence_indexes, trackableObj, image_height, image_width)

                        #write cropped images to disk
                        ime.writeThumbnailToDisk(thumbnail_list, fileName, inputQueueObj._frameNum, imgNum, saveDir)

            except Exception as e:
                print(e)

    fps.stop()
