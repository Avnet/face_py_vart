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
# python avnet_face_landmark.py [--input 0] [--detthreshold 0.55] [--nmsthreshold 0.35]

import numpy as np
import argparse
import imutils
import time
import cv2
import os, errno

from imutils.video import FPS

from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facelandmark import FaceLandmark
import runner
import xir.graph
import pathlib
import xir.subgraph

def get_subgraph (g):
  sub = []
  root = g.get_root_subgraph()
  sub = [ s for s in root.children
          if s.metadata.get_attr_str ("device") == "DPU"]
  return sub


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help = "input camera identifier (default = 0)")
ap.add_argument("-d", "--detthreshold", required=False,
	help = "face detector softmax threshold (default = 0.55)")
ap.add_argument("-n", "--nmsthreshold", required=False,
	help = "face detector NMS threshold (default = 0.35)")
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
densebox_elf = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.elf"
densebox_graph = xir.graph.Graph.deserialize(pathlib.Path(densebox_elf))
densebox_subgraphs = get_subgraph(densebox_graph)
assert len(densebox_subgraphs) == 1 # only one DPU kernel
densebox_dpu = runner.Runner(densebox_subgraphs[0],"run")
dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
dpu_face_detector.start()

# Initialize Vitis-AI/DPU based face landmark
landmark_elf = "/usr/share/vitis_ai_library/models/face_landmark/face_landmark.elf"
landmark_graph = xir.graph.Graph.deserialize(pathlib.Path(landmark_elf))
landmark_subgraphs = get_subgraph(landmark_graph)
assert len(landmark_subgraphs) == 1 # only one DPU kernel
landmark_dpu = runner.Runner(landmark_subgraphs[0],"run")
dpu_face_landmark = FaceLandmark(landmark_dpu)
dpu_face_landmark.start()

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

# loop over the frames from the video stream
while True:
	# Capture image from camera
	ret,frame = cam.read()

	# Vitis-AI/DPU based face detector
	faces = dpu_face_detector.process(frame)
	#print(faces)

	# loop over the faces
	for i,(left,top,right,bottom) in enumerate(faces): 

		# draw a bounding box surrounding the object so we can
		# visualize it
		cv2.rectangle( frame, (left,top), (right,bottom), (0,255,0), 2)

		# extract the face ROI
		startX = int(left)
		startY = int(top)
		endX   = int(right)
		endY   = int(bottom)
		#print( startX, endX, startY, endY )
		face = frame[startY:endY, startX:endX]

		# extract face landmarks
		landmarks = dpu_face_landmark.process(face)

		# draw landmarks
		for i in range(5):
			x = int(landmarks[i,0] * (endX-startX))
			y = int(landmarks[i,1] * (endY-startY))
			cv2.circle( face, (x,y), 3, (255,255,255), 2)

	# Display the processed image
	cv2.imshow("Face Detection with Landmarks", frame)
	key = cv2.waitKey(1) & 0xFF

	# Update the FPS counter
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))

# Stop the face detector
dpu_face_detector.stop()
del densebox_dpu
dpu_face_landmark.stop()
del landmark_dpu

# Cleanup
CV2.DEstroyAllWindows()
