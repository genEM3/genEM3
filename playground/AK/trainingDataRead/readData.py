import wkskel
import os
import numpy as np
from genEM3.data.wkwdata import WkwData
from matplotlib import pyplot as plt

# Read the nml and print some basic properties
nmlDir = '/gaba/u/alik/code/genEM3/playground/AK/trainingDataRead'
nmlName = 'artefact_trainingData.nml'
nmlPath = os.path.join(nmlDir, nmlName)
skel = wkskel.Skeleton(nmlPath)

# Get the coordinate of the nodes as numpy array
coordsList = [skel.nodes[x].position.values for x in range(skel.num_trees())]
# Concatenate into a single array of coordinates
coordArray = np.vstack(coordsList)
assert coordArray.shape == (111, 3)


def bboxFromCenter(center, dims):
    """Returns the 2D bounding box from center and dims arrays"""
    # input should be numpy arrays
    assert type(center) is np.ndarray and type(dims) is np.ndarray
    topLeft = center - dims/2
    # make sure it is only a single slice (3rd dim size = 0)
    return np.hstack([topLeft, dims[0:2], np.ones(1)]).astype(int)

# Get the bounding boxes of each debris location
curDims = np.array([140, 140, 0])
bboxFromCenterFixedDim = lambda coord: bboxFromCenter(coord,curDims)
bboxes = np.apply_along_axis(bboxFromCenterFixedDim, 1, coordArray)
# read the wkwdata into a numpy array
wkwDir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'
readWk = lambda bbox: WkwData.wkw_read(wkwDir, bbox)
images = np.apply_along_axis(readWk, 1, bboxes).squeeze(4)
for i in range(111):
    print(f'bbox {i}:{bboxes[i,:]}')
    plt.imshow(np.squeeze(images[i, 0, :, :]), cmap='gray')
    plt.show()
