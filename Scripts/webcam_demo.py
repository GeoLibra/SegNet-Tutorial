#!/usr/bin/env python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time


sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = './caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--weights', type=str, required=True)
# parser.add_argument('--colours', type=str, required=True)
# args = parser.parse_args()

# net = caffe.Net(args.model,
#                 args.weights,
#                 caffe.TEST)
deploy='./Example_Models/segnet_model_driving_webdemo.prototxt'
weights='./Models/SegNetModel/segnet_weights_driving_webdemo.caffemodel'
colours='./Scripts/camvid12.png'
net=caffe.Net(deploy,weights,caffe.TEST)
caffe.set_mode_gpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(colours).astype(np.uint8)

                                             
cv2.namedWindow("Input")
cv2.namedWindow("SegNet")


rval = True
start = time.time()

frame=cv2.imread('/media/hl/新加卷/SemanticSegmentation/SegNet-Tutorial/Scripts/317611_90.jpeg')
frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
input_image = frame.transpose((2,0,1))
# input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
input_image = np.asarray([input_image])
out = net.forward_all(data=input_image)

segmentation_ind = np.squeeze(net.blobs['argmax'].data)
segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
segmentation_rgb = segmentation_rgb.astype(float)/255

cv2.imwrite('./Scripts/output.jpeg',segmentation_rgb)
cv2.imshow('Input',frame)
cv2.imshow('SegNet',segmentation_rgb)
cv2.imwrite('./Scripts/output.jpeg',segmentation_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(segmentation_rgb)
# plt.savefig('./Scripts/output.jpeg')
# plt.show()