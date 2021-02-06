# USAGE
# python detect_edges_image.py  --image images/guitar.jpg
# This Code is inspired from Pyimagesearch.
# import the necessary packages
import argparse
import cv2
import os
import numpy as np
import random as rng

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--edge-detector", type=str, required=True,
# 	help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())

class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

# load our serialized edge detector from disk
# print("[INFO] loading edge detector...")
protoPath = os.path.sep.join(['./hed_model',
	"deploy.prototxt"])
modelPath = os.path.sep.join(['./hed_model',
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

# load the input image and grab its dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
image = cv2.resize(image, (int(W*.8),int(H*.8)), interpolation = cv2.INTER_AREA)
(H, W) = image.shape[:2]



blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
	mean=(104.00698793, 116.66876762, 122.67891434),
	swapRB=False, crop=False)
# set the blob as the input to the network and perform a forward pass
# to compute the edges
# print("[INFO] performing holistically-nested edge detection...")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")



contours, _ = cv2.findContours(hed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
area = {}
for i in range(len(contours)):
	ar = boundRect[i][2]*boundRect[i][3]
	area[str(i)] = ar


Keymax = max(area, key= lambda x: area[x]) 
if ((boundRect[int(Keymax)][2]*boundRect[int(Keymax)][3]) > .95 * (H*W)) :
	del area[Keymax]

# del area[Keymax]
Keymax = max(area, key= lambda x: area[x]) 
if boundRect[int(Keymax)][2]*boundRect[int(Keymax)][3] < .50 * (H*W):
		img_2 = image[(int(boundRect[int(Keymax)][1])):(int(boundRect[int(Keymax)][1]))+(int(boundRect[int(Keymax)][3])),
		(int(boundRect[int(Keymax)][0])):(int(boundRect[int(Keymax)][0]))+(int(boundRect[int(Keymax)][2]))]

		# cv2.rectangle(image, (int(boundRect[int(Keymax)][0]), int(boundRect[int(Keymax)][1])),(int(boundRect[int(Keymax)][0]+boundRect[int(Keymax)][2]), 
		# int(boundRect[int(Keymax)][1]+boundRect[int(Keymax)][3])), (0,255,0), 4)
# elif boundRect[int(Keymax)][2]*boundRect[int(Keymax)][3] < .10 * (H*W):
# 		img_2 = image[(int(boundRect[int(Keymax)][1])):(int(boundRect[int(Keymax)][1]))+(int(boundRect[int(Keymax)][3])),
# 	(int(boundRect[int(Keymax)][0])):(int(boundRect[int(Keymax)][0]))+(int(boundRect[int(Keymax)][2]))]

else:
	pass




# uncomment to visulaize the data
# cv2.imshow('Contours', image)
# cv2.imshow('croped',img_2)
# cv2.waitKey(0)

# print('./Crop_result/'+args["image"])
try:
	cv2.imwrite('./Crop_result/'+args["image"],img_2)
	print("DONE!")
except:
	cv2.imwrite('./Crop_result/'+args["image"],image)
	print("DONE!")