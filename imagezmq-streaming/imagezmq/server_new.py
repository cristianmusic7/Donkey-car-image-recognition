# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

# import the necessary packages
from imutils import build_montages
from datetime import datetime
import numpy as np
import argparse
import imutils
import cv2
import os
import tensorflow as tf

import sys
sys.path.append('./')
from imagezmq import imagezmq

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.2,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-mW", "--montageW", required=True, type=int,
# 	help="montage frame width")
# ap.add_argument("-mH", "--montageH", required=True, type=int,
# 	help="montage frame height")
# args = vars(ap.parse_args())

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# Set up camera constants
IM_WIDTH = 160
IM_HEIGHT = 128


# This is needed since the working directory is the object_detection folder.
sys.path.append('/app/research')

# Import utilites
#from utils import label_map_util
#from utils import visualization_utils as vis_util

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print("[INFO] loading model...")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join('/app/research/object_detection',MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('/app/research/object_detection','data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


on= True

print("[INFO] loading models done")

# start looping over all the frames
while on:
	# receive RPi name and frame from the RPi and acknowledge
	# the receipt
	(rpiName, frame) = imageHub.recv_image()
	imageHub.send_reply(b'OK')

	t1 = cv2.getTickCount()
        
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
	#   frame = frame1.array
	#   frame.setflags(write=1)
	frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
         [detection_boxes, detection_scores, detection_classes, num_detections],
         feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
	vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

    #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
	cv2.imshow('Object detector', frame)

    #video_writer.write(frame)
	t2 = cv2.getTickCount()

	# if the `q` key was pressed, break from the loop
	if cv2.waitKey(1) == ord('q'):
		break

		on= False

# do a bit of cleanup
cv2.destroyAllWindows()