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

	for name in trainNames:
		image = misc.imread(imagesDir + "/" + name + '/labels.png').astype(np.float32)
		image = image / 255
		labels.write_image(image)
		image = misc.imread(imagesDir + "/" + name + '/predictions.png')
		image = image / 255
		predictions.write_image(image)

convert('/isbi_predictions/2015-05-31 13-19-00/predictions/partial100/depth15/train')
convert('/isbi_predictions/2015-05-31 13-19-00/predictions/partial100/depth15/test')