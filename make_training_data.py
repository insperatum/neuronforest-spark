import numpy as np
from scipy import ndimage, misc, io
from matplotlib import pyplot
import os
import h5py

def makeFeatures(img, filename, min_idx=None, max_idx=None):
    if min_idx is None:
        min_idx = (0, 0, 0)
        max_idx = img.shape

    orders = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2], [0, 1, 1], [1, 0, 1], [0, 2, 0], [1, 1, 0], [2, 0, 0]]
    scales = [1, 2, 4]

    print("Creating features: " + filename)
    print(str(min_idx) + " to " + str(max_idx))
    #NOTE: force big-endian for use at scala end!
    features = np.empty((np.prod(max_idx-min_idx+1), len(orders) * len(scales)), dtype=">f")
    i = 0
    for scale in scales:
        print("  Scale " + str(scale))
        for o in orders:
            print("    Order " + str(o))
            f = ndimage.filters.gaussian_filter(img, scale, o)[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
            features[:, i] = f.flatten(order = 'C')
            i += 1

    print("  Saving")
    features.tofile(filename + ".raw")
    #np.savetxt(filename + ".txt", features, fmt='%.6f')
    #io.savemat(filename + ".mat", {'features':features})

def makeTargets(segTrue, filename, min_idx, max_idx):
    print("Creating targets: " + filename)
    print(str(min_idx) + " to " + str(max_idx))
    idxs = get_image_idxs(segTrue, min_idx=min_idx, max_idx=max_idx)
    targets = get_target_affinities(segTrue, idxs).astype(np.int32)
    print("  Saving")
    np.savetxt(filename + ".txt", targets, fmt='%d')

def makeDimensions(shape, filename, min_idx, max_idx):
    print("Creating dimensions: " + filename)
    print(str(min_idx) + " to " + str(max_idx))
    print("total shape = " + str(shape))
    file = open(filename + ".txt", 'w')
    file.write(" ".join([str(i) for i in shape]) + "\n")
    file.write(" ".join([str(i) for i in min_idx]) + "\n")
    file.write(" ".join([str(i) for i in max_idx]))
    file.close()

# -------------------------------------------------
def get_steps(arr):
    return tuple(np.append(np.cumprod(np.array(arr.shape)[1:][::-1])[::-1], 1))

def get_image_idxs(im, max_idx, min_idx=(0,0,0)):
    xs, ys, zs = np.ix_(range(min_idx[0], max_idx[0] + 1), range(min_idx[1], max_idx[1] + 1),
                    range(min_idx[2], max_idx[2] + 1))
    steps = get_steps(im)
    return np.array(np.unravel_index((xs * steps[0] + ys * steps[1] + zs * steps[2]).flatten(), im.shape))

def get_target_affinities(seg, idxs):
    aff = np.empty((len(idxs[0]), 3), dtype=bool)
    aff[:, 0] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[1], [0], [0]])])
    aff[:, 1] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[0], [1], [0]])])
    aff[:, 2] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[0], [0], [1]])])
    return aff


# --------------------------------------------------

def makeData(numSplit=1, margin=10, numImages=1):
    print "Loading Helmstaedter2013 data"
    Helmstaedter2013 = io.loadmat("/home/luke/data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat")
    if not os.path.exists("data"): os.mkdir("data")
    for i in range(0, numImages):
        print("Splitting im" + str(i)+ " into " + str(numSplit) + "^3 different subvolumes".format())
        bounds = Helmstaedter2013["boundingBox"][0, i]
        outer_min_idx = bounds[:, 0]
        outer_max_idx = bounds[:, 1]-1 # -1 because no affinity on faces
        box_size = (outer_max_idx - outer_min_idx + 1)/numSplit
        mainfolder = "data/im" + str(i+1)
        if not os.path.exists(mainfolder ): os.mkdir(mainfolder )
        mainfolder  = mainfolder  + "/split_" + str(numSplit)
        if not os.path.exists(mainfolder ): os.mkdir(mainfolder )
        for box_x in range(numSplit):
            for box_y in range(numSplit):
                for box_z in range(numSplit):
                    print("-------------\nCreating sub-volume " + str(box_x) + ", " + str(box_y) + ", " + str(box_z))
                    box_offset = box_size * [box_x, box_y, box_z]
                    folder = mainfolder + "/" + str(box_x) + str(box_y) + str(box_z)
                    if not os.path.exists(folder): os.mkdir(folder)

                    box_min = outer_min_idx + box_offset
                    box_max = box_min + box_size-1
                    box_min_margin = box_min - margin
                    box_max_margin = box_max + margin
                    box_min_relative = [margin, margin, margin]
                    box_max_relative = margin + box_size-1
                    shape = box_max_margin - box_min_margin + 1

                    if not os.path.exists(folder): os.mkdir(folder)
                    makeTargets(Helmstaedter2013["segTrue"][0, i], folder + "/targets", box_min, box_max)
                    makeFeatures(Helmstaedter2013["im"][0, i], folder + "/features", box_min_margin, box_max_margin)
                    makeDimensions(shape, folder + "/dimensions",  box_min_relative, box_max_relative)

makeData(2)