######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author dhq 		 # Date: 9/12/18
# removed usb camera

# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from the Picamera. 

## Based on code from Evan Juras # Date: 4/15/1
## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py



# Import packages
import os
import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# Set up camera constants
IM_WIDTH = 160
IM_HEIGHT = 128


# This is needed since the working directory is the object_detection folder.
sys.path.append('research')

# Import utilites
#from utils import label_map_util
#from utils import visualization_utils as vis_util

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join('research/object_detection',MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('research/object_detection','data','mscoco_label_map.pbtxt')

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

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.

# Initialize Picamera and grab reference to the raw capture
#camera = PiCamera()
#camera.resolution = (IM_WIDTH,IM_HEIGHT)
#camera.framerate = 10
#rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
#rawCapture.truncate(0)

import time
import pygame
import pygame.camera
import pygame.image
from datetime import datetime, timedelta

resolution=(IM_WIDTH,IM_HEIGHT)
pygame.init()
pygame.camera.init()
l= pygame.camera.list_cameras()
print('cameras',l)
camera=pygame.camera.Camera(l[0],resolution,'RGB')
camera.start()
framerate=10

frame=None
on= True
image_d = 3
print('WebcamVideoStream loaded.. .warning camera')
time.sleep(2)

#To save video file

#path = ('/home/pi/Documents/Obje.avi')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video_writer = cv2.VideoWriter(path,-1, 1, (160,128), True)
# if not video_writer :
    # print ("!!! Failed VideoWriter: invalid parameters")
    # sys.exit(1)

#camera.start_recording('/home/pi/Documents/Obje.h264')

while on:
    start=datetime.now()
    if camera.query_image():
        snapshot= camera.get_image()
        snapshot1=pygame.transform.scale(snapshot, resolution)
        frame=pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), 90))
        if image_d ==1:
            frame = rgb2gray(frame)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

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
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

        on= False
#       rawCapture.truncate(0)


    #camera.close()
    #camera.stop_recording()

#    stop = datetime.now()
#    s=1 / framerate - (stop - start).total_seconds()
#    if s > 0:
#        time.sleep(s)

camera.stop()


cv2.destroyAllWindows()
#video_writer.release()

