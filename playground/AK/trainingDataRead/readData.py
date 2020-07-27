import wkskel
import os

import numpy as np
from matplotlib import pyplot as plt
import torch

from genEM3.data.wkwdata import WkwData
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn_1px_deep, Decoder_4_sampling_bn_1px_deep
from genEM3.inference.inference import Predictor
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
bboxFromCenterFixedDim = lambda coord: bboxFromCenter(coord, curDims)
bboxes = np.apply_along_axis(bboxFromCenterFixedDim, 1, coordArray)
# read the wkwdata into a numpy array
wkwDir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'
readWk = lambda bbox: WkwData.wkw_read(wkwDir, bbox)
images = np.apply_along_axis(readWk, 1, bboxes).squeeze(4).astype('double')
# normalize images
imagesNormal = (images-148)/36
# Create pytorch dataset from numpy array
imgT = torch.Tensor(imagesNormal)
dataset_debris = torch.utils.data.TensorDataset(imgT)
dataLoader_debris = torch.utils.data.DataLoader(dataset_debris, batch_size=5)


showExamples = False
if showExamples:
    for i in range(111):
        print(f'bbox {i}:{bboxes[i,:]}')
        plt.imshow(np.squeeze(images[i, 0, :, :]), cmap='gray')
        plt.show()

    for i, example in enumerate(dataLoader_debris):
        plt.imshow(np.squeeze(example[0].numpy()), cmap='gray')
        plt.show()

# Running model ae_v03 on the data
run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, 'datasources_distributed.json')
# setting for the clean data loader
batch_size = 5
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)
num_workers = 0
# construct clean data loader from json file
datasources = WkwData.datasources_from_json(datasources_json_path)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=datasources,
)
clean_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers)
# settings for the model to be loaded
# (Is there a way to save so that you do not need to specify model again?)
state_dict_path = os.path.join(run_root, './torch_model')
device = 'cpu'
kernel_size = 3
stride = 1
n_fmaps = 16
n_latent = 192
input_size = 140
output_size = input_size
model = AE(
    Encoder_4_sampling_bn_1px_deep(input_size, kernel_size, stride, n_fmaps, n_latent),
    Decoder_4_sampling_bn_1px_deep(output_size, kernel_size, stride, n_fmaps, n_latent))
# loading the model
checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

# predicting for clean data
predictor_clean = Predictor(
    dataloader=clean_loader,
    model=model,
    state_dict=state_dict,
    device=device,
    batch_size=batch_size,
    input_shape=input_shape,
    output_shape=output_shape)
predictor_clean.predict()

# predicting for debris
predictor_debris = Predictor(
    dataloader=dataLoader_debris,
    model=model,
    state_dict=state_dict,
    device=device,
    batch_size=batch_size,
    input_shape=input_shape,
    output_shape=output_shape)
predictor_debris.predictList()
