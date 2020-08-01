import wkskel
import os

import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn import decomposition

from genEM3.data.wkwdata import WkwData
from genEM3.data.skeleton import getAllTreeCoordinates
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn_1px_deep_convonly_skip, Decoder_4_sampling_bn_1px_deep_convonly_skip
from genEM3.inference.inference import Predictor
from genEM3.util.image import normalize, readWkwFromCenter
# seed numpy random number generator
np.random.seed(5)

# Read the nml and print some basic properties
nmlDir = '/u/alik/code/genEM3/data/'
nmlName = 'artefact_trainingData.nml'
nmlPath = os.path.join(nmlDir, nmlName)
skel = wkskel.Skeleton(nmlPath)

# Get coordinates of the debris locations
coordArray = getAllTreeCoordinates(skel)
numTrainingExamples = 600
assert coordArray.shape == (numTrainingExamples, 3)
# Get the bounding boxes of each debris location and read into numpy array
dimsForCrop = np.array([140, 140, 0])
wkwDir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'
# Read images
images = readWkwFromCenter(wkwDir, coordArray, dimsForCrop)
# Normalize
imagesNormal = normalize(images, mean=148, std=36)
# Create pytorch dataLoader from numpy array
imgT = torch.Tensor(imagesNormal)
dataset_debris = torch.utils.data.TensorDataset(imgT)
dataLoader_debris = torch.utils.data.DataLoader(dataset_debris, batch_size=5)

# Plot all the images and examples from the data loader for sanity checking
showExamples = False
if showExamples:
    for i in range(111):
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
    cache_HDD=False,
    cache_RAM=True,
)
clean_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers)
# settings for the model to be loaded
# (Is there a way to save so that you do not need to specify model again?)
state_dict_path = os.path.join(run_root, './.log/torch_model')
device = 'cpu'
kernel_size = 3
stride = 1
n_fmaps = 16
n_latent = 2048
input_size = 140
output_size = input_size
model = AE(
    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_fmaps, n_latent),
    Decoder_4_sampling_bn_1px_deep_convonly_skip(output_size, kernel_size, stride, n_fmaps, n_latent))
# loading the model
checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

# Create a dictionary to keep the hidden state of debris and clean images
TYPENAMES = ('clean', 'debris')
hidden_dict = {htype: [] for htype in TYPENAMES}
# predicting for clean data
predictor_clean = Predictor(
    dataloader=clean_loader,
    model=model,
    state_dict=state_dict,
    device=device,
    batch_size=batch_size,
    input_shape=input_shape,
    output_shape=output_shape)
hidden_dict[TYPENAMES[0]] = predictor_clean.encode()

# predicting for debris
predictor_debris = Predictor(
    dataloader=dataLoader_debris,
    model=model,
    state_dict=state_dict,
    device=device,
    batch_size=batch_size,
    input_shape=input_shape,
    output_shape=output_shape)
hidden_dict[TYPENAMES[1]] = predictor_debris.encodeList()
# Concatenate individual batches into single torch tensors
hidden_dict_cat = {key: torch.cat(val, dim=0).numpy().squeeze() for key, val in hidden_dict.items()}
# Get indices for clean vs. debris images
numSamples = [x.shape[0] for x in hidden_dict_cat.values()]
indices = [np.arange(numSamples[0]), np.arange(numSamples[0], sum(numSamples))]
# colors for plot (debris: black, clean: blue)
blackC = np.zeros((1, 3))
blueC = np.zeros((1, 3))
blueC[0, 2] = 1
colors = [blueC, blackC]
colorsForPlot = {key: value for key, value in zip(TYPENAMES, colors)}
# Generate the input
hiddenMatrix = np.concatenate((hidden_dict_cat[TYPENAMES[0]], hidden_dict_cat[TYPENAMES[1]]), axis=0)

# perform the principal component analysis using scikitlearn
pca = decomposition.PCA(n_components=2)
pca.fit(hiddenMatrix)
PCs = pca.transform(hiddenMatrix)

# Plot the PCA results
for index, label in enumerate(TYPENAMES):
    plt.scatter(PCs[indices[index], 0], PCs[indices[index], 1], c=colorsForPlot.get(label), label=label)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.legend()
plt.show()
