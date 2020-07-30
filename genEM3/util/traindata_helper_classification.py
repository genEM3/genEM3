import os
import numpy as np
from genEM3.data.wkwdata import WkwData, DataSource

import wkskel

np.random.seed(5)
run_root = os.path.dirname(os.path.abspath(__file__))

nml_path = os.path.join('..', '..', 'data', 'artefact_trainingData.nml')
json_path = os.path.join('..', '..', 'data', 'datasources_classifier_v01.json')

wkw_path = "/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1"
wkw_lims = np.asarray([21500, 15500, 0, 8000, 11000, 6000])

input_mean = 148.0
input_std = 36.0

num_samples = 300
sample_dims = (140, 140, 1)

input_path = wkw_path
target_path = wkw_path

#######################################
# Get Artifact Examples from nml
#######################################

skel = wkskel.Skeleton(nml_path)
coordsList = [skel.nodes[x].position.values for x in range(skel.num_trees())]
coordArray = np.vstack(coordsList)

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
bboxes_positive = np.apply_along_axis(bboxFromCenterFixedDim, 1, coordArray)
datasources = []
for id in range(bboxes_positive.shape[0]):
    input_bbox = [int(el) for el in list(bboxes_positive[id, :])]
    datasource = DataSource(id=str(id), input_path=input_path, input_bbox=input_bbox, input_mean=input_mean,
                            input_std=input_std, target_path=target_path, target_bbox=input_bbox, target_class=1,
                            target_binary=1)
    datasources.append(datasource)

#######################################
# Get Negative Examples from random sampling
#######################################

sample_pos_x = np.random.randint(wkw_lims[0], wkw_lims[0]+wkw_lims[3]-sample_dims[0], num_samples)
sample_pos_y = np.random.randint(wkw_lims[1], wkw_lims[1]+wkw_lims[4]-sample_dims[1], num_samples)
sample_pos_z = np.random.randint(wkw_lims[2], wkw_lims[2]+wkw_lims[5]-sample_dims[2], num_samples)

for id in range(num_samples):
    input_bbox = [int(sample_pos_x[id]), int(sample_pos_y[id]), int(sample_pos_z[id]),
                  sample_dims[0], sample_dims[1], sample_dims[2]]
    target_bbox = input_bbox
    datasource = DataSource(id=str(id+bboxes_positive.shape[0]), input_path=wkw_path, input_bbox=input_bbox,
                            input_mean=input_mean, input_std=input_std, target_path=target_path, target_bbox=input_bbox,
                            target_class=0, target_binary=1)
    datasources.append(datasource)

WkwData.datasources_to_json(datasources, json_path)
