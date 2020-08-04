import numpy as np
from genEM3.data.wkwdata import WkwData
# This is a module for all image processing related functionalities


def bboxFromCenter2D(center, dims):
    """Returns the 2D bounding box from center and dims arrays"""
    # input should be numpy arrays
    assert type(center) is np.ndarray and type(dims) is np.ndarray
    topLeft = center - dims/2
    # make sure it is only a single slice (3rd dim size = 0)
    return np.hstack([topLeft, dims[0:2], np.ones(1)]).astype(int)


def bboxesFromArray(centerArray, 
                    dims=np.array([140, 140, 0])):
    """Returns the 2D bounding box from a numpy array of coordinates and the dimensions"""
    # input should be numpy arrays
    assert type(centerArray) is np.ndarray and type(dims) is np.ndarray
    bboxFromCenterFixedDim = lambda coord: bboxFromCenter2D(coord, dims)
    bboxes = np.apply_along_axis(bboxFromCenterFixedDim, 1, centerArray)
    return bboxes


def normalize(img, mean=148, std=36):
    """ Returns the image values normalized to mean of 0 and std of 1"""
    return (img-mean)/std


def readWkwFromCenter(wkwdir, coordinates, dimensions):
    """ Returns a collection of images given their coordinate and dimensions (numpy arrays)"""
    # Get the bounding boxes from coordinates and dimensions for the cropping
    bboxes = bboxesFromArray(coordinates, dimensions)
    # read the wkwdata into a numpy array
    readWk = lambda bbox: WkwData.wkw_read(wkwdir, bbox)
    images = np.apply_along_axis(readWk, 1, bboxes).squeeze(4).astype('double')
    return images
