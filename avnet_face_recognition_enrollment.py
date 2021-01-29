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
# python avnet_face_recognition_enrollment.py [--detthreshold 0.55] [--nmsthreshold 0.35] --dataset dataset --encodings encodings.pkl --display 1

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

from imutils import paths
import pickle

from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facefeature import FaceFeature
from vitis_ai_vart.utils import get_child_subgraph_dpu


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detthreshold", required=False,
	help = "face detector softmax threshold (default = 0.55)")
ap.add_argument("-n", "--nmsthreshold", required=False,
	help = "face detector NMS threshold (default = 0.35)")
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-y", "--display", type=int, default=0,
	help="whether or not to display output frame to screen")
args = vars(ap.parse_args())

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

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	print("   {} : {}".format(name,imagePath))

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)

	# resize images to 640x480 range
	height, width = image.shape[:2]
	#print( np.float32(height)/480.0, np.float32(width)/640.0 )
	IMAGE_RESIZE = max( np.float32(height)/480.0, np.float32(width)/640.0 )
	image = cv2.resize(image,None,
                       fx=1.0/IMAGE_RESIZE,
                       fy=1.0/IMAGE_RESIZE,
                       interpolation = cv2.INTER_LINEAR)
    
        # Detect faces
	faces = dpu_face_detector.process(image)
	if len(faces) != 1:
	    print("   Skipping invalid image : Found ",len(faces)," faces")       
	    continue

	# loop over the faces
	for i,(left,top,right,bottom) in enumerate(faces): 

		# draw a bounding box surrounding the object so we can
		# visualize it
		cv2.rectangle( image, (left,top), (right,bottom), (0,255,0), 2)
		
		# extract the face ROI
		startX = int(left)
		startY = int(top)
		endX   = int(right)
		endY   = int(bottom)
		face = image[startY:endY, startX:endX]

		# extract face features
		features = dpu_face_features.process(face)

		# add each features + name to our set of known names and encodings
		knownEncodings.append(features)
		knownNames.append(name)


	# Display the processed image
	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Face Encoding", image)
		key = cv2.waitKey(1) & 0xFF
		#key = cv2.waitKey(0) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# Dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

# Stop the DPU models
dpu_face_detector.stop()
del densebox_dpu
dpu_face_features.stop()
del facerec_dpu

# Cleanup
cv2.destroyAllWindows()

