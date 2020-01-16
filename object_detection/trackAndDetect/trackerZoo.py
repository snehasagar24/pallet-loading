from object_detection.utils import visualization_utils as vis_util
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib
import cv2
import tensorflow as tf

def start_tracker(box, label, rgb, inputQueue, outputQueue):
	# construct a dlib rectangle object from the bounding box
	# coordinates and then start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)

	# loop indefinitely -- this function will be called as a daemon
	# process so we don't need to worry about joining it
	while True:
		# attempt to grab the next frame from the input queue
		rgb = inputQueue.get()

		# if there was an entry in our queue, process it
		if rgb is not None:
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the label + bounding box coordinates to the output
			# queue
			outputQueue.put((label, (startX, startY, endX, endY)))

def start_tracking(ckpt, category_index, box, rgb, inputQueue, outputQueue):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)

    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        rgb = inputQueue.get()

        # if there was an entry in our queue, process it
        if rgb is not None:
            # update the tracker and grab the position of the tracked object
            t.update(rgb)

            #get annotated image with rectangles, labels and confidence scores
            image_np,_,_,_,_ = box_detector(ckpt, category_index, rgb)

            # add the label + bounding box coordinates to the output
            # queue
            outputQueue.put(image_np)

def box_detector(ckpt, category_index, rgb):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_np, boxes, scores, classes, num_detections = detect_objects(rgb, category_index, sess, detection_graph)
    sess.close()

    return image_np, boxes, scores, classes, num_detections

def detect_objects(image_np, category_index, sess, detection_graph):
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
        line_thickness=8)

    return image_np, boxes, scores, classes, num_detections

#class TrackingProcess():

#    def __init__(self, confidence, category_index, PATH_TO_CKPT):
#        self._confidence = confidence
#        self._category_index = category_index
#        self._ckpt = PATH_TO_CKPT

#    def box_detector(self, rgb):
#        # Load a (frozen) Tensorflow model into memory.
#        detection_graph = tf.Graph()
#        with detection_graph.as_default():
#            od_graph_def = tf.GraphDef()
#            with tf.gfile.GFile(self._ckpt, 'rb') as fid:
#                serialized_graph = fid.read()
#                od_graph_def.ParseFromString(serialized_graph)
#                tf.import_graph_def(od_graph_def, name='')

#            sess = tf.Session(graph=detection_graph)

#        image_np, boxes, scores, classes, num_detections = self.detect_objects(rgb, sess, detection_graph)
#        sess.close()

#        return image_np, boxes, scores, classes, num_detections

#    def high_confidence_detections(self, boxes, scores, classes):
#        #enumerate over the scores and filter out weak detections
#        high_confidence_indexes = [i for i in range(scores.shape[1]) if scores[0][i] >= self._confidence]

#        for idx in high_confidence_indexes:
#            category_idx = classes[0][idx]
#            label = self._category_index[category_idx]["name"]
#            box = boxes[0][idx]
#            bb = (startX, startY, endX, endY) = box.astype("int")
#            score = scores[0][idx]

#            # construct a dlib rectangle object from the bounding box
#	        # coordinates and then start the correlation tracker
#            t = dlib.correlation_tracker()
#            rect = dlib.rectangle(box[0], box[1], box[2], box[3])
#            t.start_track(rgb, rect)
                

#    def manage_tracker(self, rgb, input_q, output_q):

#        image_np, boxes, scores, classes, num_detections = self.box_detector(rgb)

#        #insantiate individual trackers
#        self.high_confidence_detections(boxes,scores,classes)

#	    # loop indefinitely -- this function will be called as a daemon
#	    # process so we don't need to worry about joining it
#        while True:
#            # attempt to grab the next frame from the input queue
#            rgb = input_q.get()

#            # if there was an entry in our queue, process it
#            if rgb is not None:
#                # update the tracker and grab the position of the tracked
#                # object
#                t.update(rgb)
#                image_np, boxes, scores, classes, num_detections = self.box_detector(rgb)
#                # add the label + bounding box coordinates to the tracking queue
#                output_q.put(image_np)

#    def detect_objects(self, image_np, sess, detection_graph):
#        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#        image_np_expanded = np.expand_dims(image_np, axis=0)
#        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

#        # Each box represents a part of the image where a particular object was detected.
#        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

#        # Each score represent how level of confidence for each of the objects.
#        # Score is shown on the result image, together with the class label.
#        scores = detection_graph.get_tensor_by_name('detection_scores:0')
#        classes = detection_graph.get_tensor_by_name('detection_classes:0')
#        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#        # Actual detection.
#        (boxes, scores, classes, num_detections) = sess.run(
#            [boxes, scores, classes, num_detections],
#            feed_dict={image_tensor: image_np_expanded})

#        # Visualization of the results of a detection.
#        vis_util.visualize_boxes_and_labels_on_image_array(
#            image_np,
#            np.squeeze(boxes),
#            np.squeeze(classes).astype(np.int32),
#            np.squeeze(scores),
#            category_index,
#            use_normalized_coordinates=True,
#            line_thickness=8)

#        return image_np, boxes, scores, classes, num_detections