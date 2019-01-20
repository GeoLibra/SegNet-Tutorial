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
import os
from os.path import join
import shutil
from PIL import Image
from pymongo import MongoClient
from progressbar import ProgressBar
import multiprocessing
sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = './caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python')
import caffe
client = MongoClient('127.0.0.1', 27017)
db = client.street_view
# Import arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--weights', type=str, required=True)
# parser.add_argument('--colours', type=str, required=True)
# args = parser.parse_args()

# net = caffe.Net(args.model,
#                 args.weights,
#                 caffe.TEST)

def start_process():
	deploy='./Example_Models/segnet_model_driving_webdemo.prototxt'
	weights='./Models/SegNetModel/segnet_weights_driving_webdemo.caffemodel'
	colours='./Scripts/camvid12.png'
	caffe.set_mode_gpu()
	net=caffe.Net(deploy,weights,caffe.TEST)
def getClass(arr):
	Sky = [128,128,128]
	Building = [128,0,0]
	Pole = [192,192,128]
	Road_marking = [255,69,0]
	Road = [128,64,128]
	Pavement = [60,40,222]
	Tree = [128,128,0]
	SignSymbol = [192,128,128]
	Fence = [64,64,128]
	Car = [64,0,128]
	Pedestrian = [64,64,0]
	Bicyclist = [0,128,192]
	Unlabelled = [0,0,0]
	classArr=[Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled]
	className=['Sky', 'Building', 'Pole', 'Road_marking','Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Unlabelled']
	
	return className[classArr.index(arr)]
def getClassArea(pic,pic_id):
	w=pic.shape
	result={'pic_id':pic_id}
	for i in range(w[0]):
		for j in range(w[1]):
			className=getClass([pic[i][j][0],pic[i][j][1],pic[i][j][2]])
			if className in result.keys():
				result[className]+=1
			else:
				result[className]=1
	cursor = db.sv.insert(result)

def segPicture(input_pic):
	deploy='./Example_Models/segnet_model_driving_webdemo.prototxt'
	weights='./Models/SegNetModel/segnet_weights_driving_webdemo.caffemodel'
	colours='./Scripts/camvid12.png'
	caffe.set_mode_gpu()
	net=caffe.Net(deploy,weights,caffe.TEST)
	filename = os.path.basename(input_pic)
	result='/media/hl/mydata/segm2/'
	output_pic=result+filename
	pic_id=filename.split('.')[0]
	input_shape = net.blobs['data'].data.shape
	output_shape = net.blobs['argmax'].data.shape
	label_colours = cv2.imread(colours).astype(np.uint8)
	# label_colours_plt=plt.imread(colours).astype(np.uint8)
	# print(label_colours)
	# print(label_colours_plt)	
	# cv2.namedWindow("Input")
	# cv2.namedWindow("SegNet")
	rval = True
	start = time.time()

	frame=cv2.imread(input_pic)

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
	# output = np.transpose(segmentation_rgb, (2,0,1))
	output = segmentation_rgb[:,:,(2,1,0)]
	getClassArea(output,pic_id)
	segmentation_rgb = segmentation_rgb.astype(float)/255
	output_pic='/media/hl/mydata/segm2/'+filename
	cv2.imwrite(output_pic,segmentation_rgb)


# segPicture('/media/hl/mydata/SemanticSegmentation/SegNet-Tutorial/Scripts/v2.png',
# '/media/hl/mydata/SemanticSegmentation/SegNet-Tutorial/Scripts/v2_mask.png')


def m(files):
	result='/media/hl/mydata/segm2'
	progress=ProgressBar()
	for name in progress(files):
		input_pic=join(root,name)
		segPicture(input_pic,join(result,name))

if __name__=="__main__":
	path='/media/hl/mydata/photos'
	
	full_list=[]
	for root,dirs,filenames in os.walk(path):
			for name in filenames:
					full_list.append(join(root,name))
	# n_total=len(full_list)
	# n_processes=3
	# length=n_total/n_processes
	# indices=[int(round(i*length)) for i in range(n_processes+1)]
	# # 生成每个进程要处理的子文件列表
	# sublists=[full_list[indices[i]:indices[i+1]] for i in range(n_processes)]
	# # 生成进程
	# processes=[Process(target=m,args=(x,)) for x in sublists]
	# # 并行处理
	# pool_size=multiprocessing.cpu_count()*2
	pool_size=3
	pool=multiprocessing.Pool(processes=pool_size,)
	pool_outputs=pool.map(segPicture,full_list)
	pool.close()
	pool.join()