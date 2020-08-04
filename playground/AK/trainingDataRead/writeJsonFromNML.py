import os
import wkskel
import numpy as np
from genEM3.util.path import getDataDir, getMag8DatasetDir
from genEM3.data.wkwdata import DataSource, WkwData
from genEM3.data.skeleton import getAllTreeCoordinates
from genEM3.util.image import bboxesFromArray
nmlPath = os.path.join(getDataDir(), 'artefact_trainingData.nml')
skel = wkskel.Skeleton(nmlPath)

# Get coordinates of the debris locations
coordArray = getAllTreeCoordinates(skel)
numTrainingExamples = 600
assert coordArray.shape == (numTrainingExamples, 3)

# Get the bounding boxes of each debris location and read into numpy array
dimsForCrop = np.array([140, 140, 0])
bboxes_debris = bboxesFromArray(coordArray, dimsForCrop)

# The clean bounding boxes (inspected by me)
bboxes_clean = [[24310, 22880, 640, 140, 140, 50],
                [24868, 20876, 1731, 140, 140, 50],
                [30163, 16682, 662, 140, 140, 50],
                [25985, 17030, 2768, 140, 140, 50],
                [21980, 15643, 2705, 140, 140, 50],
                [27701, 20539, 2881, 140, 140, 50],
                [21052, 16640, 3107, 140, 140, 50],
                [19631, 15376, 3267, 140, 140, 50],
                [24568, 15582, 3365, 140, 140, 50],
                [24761, 15838, 3341, 140, 140, 50],
                [29011, 18583, 4956, 140, 140, 50],
                [29723, 22197, 5076, 140, 140, 50],
                [29948, 16404, 6123, 140, 140, 50]]
# create a list of the data sources
dataSources = []
# Append debris locations
for idx, curBbox in enumerate(bboxes_debris):
    # convert bbox to normal python list and integer. numpy arrays are not serializable
    curBbox = [int(num) for num in curBbox]
    curSource = DataSource(id=str(idx),
                           input_path=getMag8DatasetDir(),
                           input_bbox=curBbox,
                           input_mean=148.0,
                           input_std=36.0,
                           target_path=getMag8DatasetDir(),
                           target_bbox=curBbox,
                           target_class=1.0,
                           target_binary=1)
    dataSources.append(curSource)
# Append clean locations
for idx, curBbox in enumerate(bboxes_clean):
    # The initial 600 Indices are taken by the debris locations
    idx = idx + numTrainingExamples
    curSource = DataSource(id=str(idx),
                           input_path=getMag8DatasetDir(),
                           input_bbox=curBbox,
                           input_mean=148.0,
                           input_std=36.0,
                           target_path=getMag8DatasetDir(),
                           target_bbox=curBbox,
                           target_class=0.0,
                           target_binary=1)
    dataSources.append(curSource)
# write to JSON file
jsonPath = os.path.join(getDataDir(), 'debris_clean_datasource.json')
WkwData.datasources_to_json(dataSources, jsonPath)
