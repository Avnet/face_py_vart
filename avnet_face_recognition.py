'''
Copyright 2020 Avnet Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# USAGE
# python avnet_face_recognition.py [--input 0] [--detthreshold 0.55] [--nmsthreshold 0.35] --encodings encodings.pkl

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys
import argparse

from imutils.video import VideoStream
from imutils.video import FPS
import pickle

from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facefeature import FaceFeature
from vitis_ai_vart.utils import get_child_subgraph_dpu


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help = "input camera identifier (default = 0)")
ap.add_argument("-d", "--detthreshold", required=False,
	help = "face detector softmax threshold (default = 0.55)")
ap.add_argument("-n", "--nmsthreshold", required=False,
	help = "face detector NMS threshold (default = 0.35)")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
args = vars(ap.parse_args())

if not args.get("input",False):
  inputId = 0
else:
  inputId = int(args["input"])
print('[INFO] input camera identifier = ',inputId)

if not args.get("detthreshold",False):
  detThreshold = 0.55
else:
  detThreshold = float(args["detthreshold"])
print('[INFO] face detector - softmax threshold = ',detThreshold)

if not args.get("nmsthreshold",False):
  nmsThreshold = 0.35
else:
  nmsThreshold = float(args["nmsthreshold"])
print('[INFO] face detector - NMS threshold = ',nmsThreshold)


# Initialize Vitis-AI/DPU based face detector
densebox_xmodel = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel"
densebox_graph = xir.Graph.deserialize(densebox_xmodel)
densebox_subgraphs = get_child_subgraph_dpu(densebox_graph)
assert len(densebox_subgraphs) == 1 # only one DPU kernel
densebox_dpu = vart.Runner.create_runner(densebox_subgraphs[0],"run")
dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
dpu_face_detector.start()

# Initialize Vitis-AI/DPU based face features
#facerec_xmodel = "/usr/share/vitis_ai_library/models/facerec_resnet20/facerec_resnet20.xmodel"
facerec_xmodel = "/usr/share/vitis_ai_library/models/facerec_resnet64/facerec_resnet64.xmodel"
facerec_graph = xir.Graph.deserialize(facerec_xmodel)
facerec_subgraphs = get_child_subgraph_dpu(facerec_graph)
assert len(facerec_subgraphs) == 1 # only one DPU kernel
facerec_dpu = vart.Runner.create_runner(facerec_subgraphs[0],"run")
dpu_face_features = FaceFeature(facerec_dpu)
dpu_face_features.start()

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
faceDescriptorsList = data["encodings"]
faceDescriptorsNdarray = np.asarray(faceDescriptorsList, dtype=np.float64)
faceDescriptorsNdarray = np.squeeze(faceDescriptorsNdarray,axis=(1,))
faceDescriptorsEnrolled = faceDescriptorsNdarray

# Initialize the camera input
print("[INFO] starting camera input ...")
cam = cv2.VideoCapture(inputId)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not (cam.isOpened()):
    print("[ERROR] Failed to open camera ", inputId )
    exit()

# start the FPS counter
fps = FPS().start()

# writer
writer = None

# loop over frames from the video file stream
while True:
        # Capture image from camera
	ret,frame = cam.read()
	
	faces = dpu_face_detector.process(frame)
	names = []

	# loop over the faces
	for i,(left,top,right,bottom) in enumerate(faces): 

		# draw a bounding box surrounding the object so we can
		# visualize it
		#cv2.rectangle( frame, (left,top), (right,bottom), (0,255,0), 2)
		
		# extract the face ROI
		startX = int(left)
		startY = int(top)
		endX   = int(right)
		endY   = int(bottom)
		face = frame[startY:endY, startX:endX]

		# extract face features
		features = dpu_face_features.process(face)

		name = "Unknown"

		faceDescriptorNdarray = features

		# Calculate Euclidean distances between face descriptor calculated on face dectected
		# in current frame with all the face descriptors we calculated while enrolling faces
		distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1)
		#print(distances)

		# Calculate minimum distance and index of this face
		argmin = np.argmin(distances)  # index
		minDistance = distances[argmin]  # minimum distance

		name = data["names"][argmin]
		print(name)

		# update the list of names
		names.append(name)


	# loop over the faces
	for ((left,top,right,bottom),name) in zip(faces,names): 

		top = int(top)
		right = int(right)
		bottom = int(bottom)
		left = int(left)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces to disk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# Update the FPS counter
	fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))

# Stop the DPU models
dpu_face_detector.stop()
del densebox_dpu
dpu_face_features.stop()
del facerec_dpu

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()

# Cleanup
cv2.destroyAllWindows()

