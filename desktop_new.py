# desktop.py

import asyncio
import aiohttp
import cv2
import json
import numpy as np
import argparse
from imutils.video import FPS
import imutils
from rtcbot import RTCConnection, Gamepad, CVDisplay

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] loading model done")

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

CONSIDER = set(["dog", "person", "car"])
objCount = {obj: 0 for obj in CONSIDER}

disp = CVDisplay()
#g = Gamepad()
conn = RTCConnection()
fps = FPS().start()

async def fpsCheck():
    while True:
        await asyncio.sleep(5)
        if fps:
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


@conn.video.subscribe
def onFrame(frame):
    # Show a 4x larger image so that it is easy to see
    #resized = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    #resized = imutils.resize(frame, width=400)
    fps.update()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

	# reset the object count for each object in the CONSIDER set
    objCount = {obj: 0 for obj in CONSIDER}

	# loop over the detections
    for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
        confidence = detections[0, 0, i, 2]
    
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
        if confidence > args["confidence"]:
        	# extract the index of the class label from the
        	# detections
        	idx = int(detections[0, 0, i, 1])
        	# check to see if the predicted class is in the set of
        	# classes that need to be considered
        	if CLASSES[idx] in CONSIDER:
        		# increment the count of the particular object
        		# detected in the frame
        		objCount[CLASSES[idx]] += 1
        		# compute the (x, y)-coordinates of the bounding box
        		# for the object
        		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        		(startX, startY, endX, endY) = box.astype("int")
        		# draw the bounding box around the detected object on
        		# the frame
        		cv2.rectangle(frame, (startX, startY), (endX, endY),
        			(255, 0, 0), 2)
        # draw the object count on the frame
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
    cv2.putText(frame, label, (10, h - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
    
    #cv2.imshow("Home pet location monitor ({})".format(i), frame)
    disp.put_nowait(frame)

async def connect():
    localDescription = await conn.getLocalDescription()
    async with aiohttp.ClientSession() as session:
        async with session.post(
                "http://192.168.0.3:8080/connect", data=json.dumps(localDescription)
        ) as resp:
            response = await resp.json()
            await conn.setRemoteDescription(response)
    # Start sending gamepad controls
    #g.subscribe(conn)

asyncio.ensure_future(fpsCheck())
asyncio.ensure_future(connect())
try:
    asyncio.get_event_loop().run_forever()
finally:
    fps.stop()
    conn.close()
    disp.close()
    #g.close()
                    