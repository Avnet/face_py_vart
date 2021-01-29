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


from ctypes import *
import cv2
import os
import threading
import time
import sys
import numpy as np
from numpy import float32
import math

import vart
#from utils import get_child_subgraph_dpu
  
def time_it(msg,start,end):
    print("[INFO] {} took {:.8} seconds".format(msg,end-start))


class FaceFeature():
#  def __init__(self, dpu_xmodel):
#    #"""Create Runner"""
#    dpu_graph = xir.Graph.deserialize(dpu_xmodel)
#    dpu_subgraphs = get_child_subgraph_dpu(dpu_graph)
#    assert len(dpu_subgraphs) == 1 # only one DPU kernel
#    print("[INFO] FaceFeature dpu_xmodel=",dpu_xmodel)
#    dpu = vart.Runner.create_runner(dpu_subgraphs[0],"run")


  def __init__(self, dpu):

    self.dpu = dpu
    
    self.inputTensors = []
    self.outputTensors = []
    self.inputChannels = []
    self.inputHeight = []
    self.inputWidth = []
    self.inputShape = []
    self.outputSize = []
    self.outputShape = []

  def start(self):

    dpu = self.dpu
    #print("[INFO] facefeature runner=",dpu)

    inputTensors = dpu.get_input_tensors()
    #print("[INFO] inputTensors=",inputTensors)
    outputTensors = dpu.get_output_tensors()
    #print("[INFO] outputTensors=",outputTensors)
    
    inputHeight = inputTensors[0].dims[1]
    inputWidth = inputTensors[0].dims[2]
    inputChannels = inputTensors[0].dims[3]
    #print("[INFO] input tensor : format=NHWC, Height=",inputHeight," Width=",inputWidth,", Channels=", inputChannels)
    
    #print("[INFO] outputTensors[0]=",outputTensors[0])
    outputSize = outputTensors[0].dims[1]
    #print("[INFO] output[0] tensor : size=",outputSize)

    inputShape = (1,inputHeight,inputWidth,inputChannels)
    #print("[INFO] inputShape=",inputShape)
    outputShape = (1,outputSize)
    #print("[INFO] outputShape=",outputShape)

    self.inputTensors = inputTensors
    self.outputTensors = outputTensors
    self.inputChannels = inputChannels
    self.inputHeight = inputHeight
    self.inputWidth = inputWidth
    self.inputShape = inputShape
    self.outputSize = outputSize
    self.outputShape = outputShape

  def process(self,img):
    #print("[INFO] facefeature process")

    dpu = self.dpu
    #print("[INFO] facefeature runner=",dpu)

    inputChannels = self.inputChannels
    inputHeight = self.inputHeight
    inputWidth = self.inputWidth
    inputShape = self.inputShape
    outputSize = self.outputSize
    outputShape = self.outputShape

    imgHeight = img.shape[0]
    imgWidth  = img.shape[1]
    scale_h = imgHeight / inputHeight
    scale_w = imgWidth / inputWidth
    
    """ Image pre-processing """
    #print("[INFO] process - pre-processing - resize ")
    resize_img = cv2.resize(img,(inputWidth,inputHeight))
    #print("[INFO] process - pre-processing - convert to float ")
    input_image = resize_img.astype(np.float)
    #print("[INFO] process - pre-processing - normalize (-128.0) ")
    input_image = input_image - 128.0
    #print("[INFO] process - pre-processing - scale (*0.0078125) ")
    input_image = input_image * 0.0078125

    """ Prepare input/output buffers """
    #print("[INFO] process - prep input buffer ")
    inputData = []
    inputData.append(np.empty((inputShape),dtype=np.float32,order='C'))
    inputImage = inputData[0]
    inputImage[0,...] = input_image

    #print("[INFO] process - prep output buffer ")
    outputData = []
    outputData.append(np.empty((outputShape),dtype=np.float32,order='C'))

    """ Execute model on DPU """
    #print("[INFO] process - execute ")
    job_id = dpu.execute_async( inputData, outputData )
    dpu.wait(job_id)

    """ Retrieve output results """    
    #print("[INFO] process - get output ")
    OutputData = outputData[0].reshape(1,outputSize)
    #print(OutputData)
    features = np.reshape( OutputData, (-1, 512) )

    return features

  def stop(self):
    #"""Destroy Runner"""
    del self.dpu
	
    self.dpu = []
    self.inputTensors = []
    self.outputTensors = []
    self.tensorFormat = []
    self.input0Channels = []
    self.inputHeight = []
    self.inputWidth = []
    self.inputShape = []
    self.outputSize = []
    self.outputShape = []


