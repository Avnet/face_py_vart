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
# python avnet_face_detection_mt.py [--input 0] [--detthreshold 0.55] [--nmsthreshold 0.35] [--threads 4]

import numpy as np
import argparse
import imutils
import time
import cv2
import os, errno
import sys
import threading
import queue

from imutils.video import FPS

from vitis_ai_vart.facedetect import FaceDetect
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

global bQuit


def taskCapture(inputId,queueIn):

    global bQuit

    #print("[INFO] taskCapture : starting thread ...")

    # Start the FPS counter
    fpsIn = FPS().start()

    # Initialize the camera input
    print("[INFO] taskCapture : starting camera input ...")
    cam = cv2.VideoCapture(inputId)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not (cam.isOpened()):
        print("[ERROR] taskCapture : Failed to open camera ", inputId )
        exit()

    while not bQuit:
        # Capture image from camera
        ret,frame = cam.read()

        # Update the FPS counter
        fpsIn.update()

        # Push captured image to input queue
        queueIn.put(frame)

    # Stop the timer and display FPS information
    fpsIn.stop()
    print("[INFO] taskCapture : elapsed time: {:.2f}".format(fpsIn.elapsed()))
    print("[INFO] taskCapture : elapsed FPS: {:.2f}".format(fpsIn.fps()))

    #print("[INFO] taskCapture : exiting thread ...")


def taskWorker(worker,dpu,detThreshold,nmsThreshold,queueIn,queueOut):

    global bQuit

    #print("[INFO] taskWorker[",worker,"] : starting thread ...")

    # Start the face detector
    dpu_face_detector = FaceDetect(dpu,detThreshold,nmsThreshold)
    dpu_face_detector.start()

    while not bQuit:
        # Pop captured image from input queue
        frame = queueIn.get()

        # Vitis-AI/DPU based face detector
        faces = dpu_face_detector.process(frame)

        # loop over the faces
        for i,(left,top,right,bottom) in enumerate(faces): 

            # draw a bounding box surrounding the object so we can
            # visualize it
            cv2.rectangle( frame, (left,top), (right,bottom), (0,255,0), 2)


        # Push processed image to output queue
        queueOut.put(frame)

    # Stop the face detector
    dpu_face_detector.stop()

    # workaround : to ensure other worker threads stop, 
    #              make sure input queue is not empty 
    queueIn.put(frame)

    #print("[INFO] taskWorker[",worker,"] : exiting thread ...")

def taskDisplay(queueOut):

    global bQuit

    #print("[INFO] taskDisplay : starting thread ...")

    # Start the FPS counter
    fpsOut = FPS().start()

    while not bQuit:
        # Pop processed image from output queue
        frame = queueOut.get()

        # Display the processed image
        cv2.imshow("Face Detection", frame)

        # Update the FPS counter
        fpsOut.update()

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Trigger all threads to stop
    bQuit = True

    # Stop the timer and display FPS information
    fpsOut.stop()
    print("[INFO] taskDisplay : elapsed time: {:.2f}".format(fpsOut.elapsed()))
    print("[INFO] taskDisplay : elapsed FPS: {:.2f}".format(fpsOut.fps()))

    # Cleanup
    cv2.destroyAllWindows()

    #print("[INFO] taskDisplay : exiting thread ...")


def main(argv):

    global bQuit
    bQuit = False

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False,
        help = "input camera identifier (default = 0)")
    ap.add_argument("-d", "--detthreshold", required=False,
        help = "face detector softmax threshold (default = 0.55)")
    ap.add_argument("-n", "--nmsthreshold", required=False,
        help = "face detector NMS threshold (default = 0.35)")
    ap.add_argument("-t", "--threads", required=False,
        help = "number of worker threads (default = 4)")
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

    if not args.get("threads",False):
        threads = 4
    else:
        threads = int(args["threads"])
    print('[INFO] number of worker threads = ', threads )

    # Initialize VART API
    densebox_elf = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.elf"
    densebox_graph = xir.graph.Graph.deserialize(pathlib.Path(densebox_elf))
    densebox_subgraphs = get_subgraph(densebox_graph)
    assert len(densebox_subgraphs) == 1 # only one DPU kernel
    all_dpu_runners = [];
    for i in range(int(threads)):
        all_dpu_runners.append(runner.Runner(densebox_subgraphs[0], "run"));

    # Init synchronous queues for inter-thread communication
    queueIn = queue.Queue()
    queueOut = queue.Queue()

    # Launch threads
    threadAll = []
    tc = threading.Thread(target=taskCapture, args=(inputId,queueIn))
    threadAll.append(tc)
    for i in range(threads):
        tw = threading.Thread(target=taskWorker, args=(i,all_dpu_runners[i],detThreshold,nmsThreshold,queueIn,queueOut))
        threadAll.append(tw)
    td = threading.Thread(target=taskDisplay, args=(queueOut,))
    threadAll.append(td)
    for x in threadAll:
        x.start()

    # Wait for all threads to stop
    for x in threadAll:
        x.join()

    # Cleanup VART API
    del dpu


if __name__ == "__main__":
    main(sys.argv)
