from scipy import misc
import os
import numpy as np
from libtiff import TIFF
import re

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def convert(imagesDir):
	if(os.path.isfile(imagesDir + "/labels.tif")): os.remove(imagesDir + "/labels.tif")
	if(os.path.isfile(imagesDir + "/predictions.tif")): os.remove(imagesDir + "/predictions.tif")
	

	trainNames = sorted(os.listdir(imagesDir), key=natural_key)

	labels = TIFF.open(imagesDir + "/labels.tif", "w")
	predictions = TIFF.open(imagesDir + "/predictions.tif", "w")

	maxlabel = -999999
	maxprediction = -999999
	minlabel = 999999
	minprediction = 999999
	for name in trainNames:
		image = misc.imread(imagesDir + "/" + name + '/labels.png').astype(np.float32)
		maxlabel = max(maxlabel, image.max())
		minlabel = min(minlabel, image.min())
		image = misc.imread(imagesDir + "/" + name + '/labels.png').astype(np.float32)
		maxprediction = max(maxprediction, image.max())
		minprediction = min(minprediction, image.min())	
	
	print("Label values: [" + str(minlabel) + ", " + str(maxlabel) + "]")
	print("Prediction values: [" + str(minprediction) + ", " + str(maxprediction) + "]")

	for name in trainNames:
		image = misc.imread(imagesDir + "/" + name + '/labels.png').astype(np.float32)
		image = (image - minlabel) / (maxlabel - minlabel)
		labels.write_image(image)
		image = misc.imread(imagesDir + "/" + name + '/predictions.png')
		image = (image - minprediction) / (maxprediction - minprediction)
		predictions.write_image(image)

convert('/isbi_predictions/2015-05-29 09-10-02/predictions/partial10/train')