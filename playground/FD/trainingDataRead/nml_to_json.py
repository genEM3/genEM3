import wkskel
import os

import numpy as np
from matplotlib import pyplot as plt
import torch

from genEM3.data.wkwdata import WkwData
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn_1px_deep, Decoder_4_sampling_bn_1px_deep
from genEM3.inference.inference import Predictor
# seed numpy random number generator
np.random.seed(5)

# Read the nml and print some basic properties
nmlName = 'artefact_trainingData.nml'
skel = wkskel.Skeleton(nmlName)

# Get the coordinate of the nodes as numpy array
coordsList = [skel.nodes[x].position.values for x in range(skel.num_trees())]
# Concatenate into a single array of coordinates
coordArray = np.vstack(coordsList)
assert coordArray.shape == (115, 3)


def bboxFromCenter(center, dims):
    """Returns the 2D bounding box from center and dims arrays"""
    # input should be numpy arrays
    assert type(center) is np.ndarray and type(dims) is np.ndarray
    topLeft = center - dims/2
    # make sure it is only a single slice (3rd dim size = 0)
    return np.hstack([topLeft, dims[0:2], np.ones(1)]).astype(int)


# Get the bounding boxes of each debris location
curDims = np.array([140, 140, 0])
bboxFromCenterFixedDim = lambda coord: bboxFromCenter(coord, curDims)
bboxes = np.apply_along_axis(bboxFromCenterFixedDim, 1, coordArray)
# read the wkwdata into a numpy array
wkwDir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'
