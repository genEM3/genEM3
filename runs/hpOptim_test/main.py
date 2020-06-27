# TODO: Add hyperopt and ray to requirements file
# generic imports
import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

# ray[tune] imports
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
# genEM3 imports
from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn, Decoder_4_sampling_bn
from genEM3.training.training import Trainer
from genEM3.training.hyperParamOptim import trainable
from genEM3.util import gpu

# Parameters for loading the training data
run_root = '/tmpscratch/alik/runlogs/AE_2d/hpOptim_test'
# create directory if not preset
if not(os.path.isdir(run_root)):
    os.mkdir(run_root)

datasources_json_path = os.path.join(run_root, 'datasources.json')
input_shape = (302, 302, 1)
output_shape = (302, 302, 1)
# Read parameters for the data into a list of named tuples
data_sources = WkwData.datasources_from_json(datasources_json_path)
data_split = DataSplit(train=0.7, validation=0.2, test=0.1)
cache_RAM = True
cache_HDD = True
cache_root = os.path.join(run_root, '.cache/')
if not(os.path.isdir(cache_root)):
    os.mkdir(cache_root)
batch_size = 32
num_workers = 4
# create dataset from the data source json parameters
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD,
    cache_HDD_root=cache_root
)

# Create the pytorch data loader objects
train_sampler = SubsetRandomSampler(dataset.data_train_inds)
validation_sampler = SubsetRandomSampler(dataset.data_validation_inds)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
    collate_fn=dataset.collate_fn)
validation_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=validation_sampler,
    collate_fn=dataset.collate_fn)

train_loader.dataset[0]
device = 'cpu'
if device == 'cuda':
    # Get the empty gpu
    gpu.get_empty_gpu()

# /tmp is not accessible on GABA use the following dir:
ray.init(temp_dir='/tmpscratch/alik/runlogs/ray/')

# Log uniform function
def lognuniform(low=0, high=1, base=np.e):
    size = 1
    return int(np.power(base, np.random.uniform(low, high, size)))

# random search
# space = {"lr": tune.loguniform(1e-6, 0.1), "momentum": tune.loguniform(0.8, 0.9999),
#          "n_latent": tune.choice(list(range(100,10000))), # tune.sample_from(lambda _:lognuniform(2, 4, 10)),
#          "n_fmaps": tune.choice(list(range(4, 16))),
#          "validation_loader": validation_loader,
#          "train_loader": train_loader}

space = {"lr": 0.01, 
         "momentum": 0.999,
         "n_latent": 100,
         "n_fmaps": 8,
         "validation_loader": validation_loader,
         "train_loader": train_loader}
thisTrainable = trainable(space)
thisTrainable._train()

# analysis = tune.run(trainable,
#                     config=space,
#                     num_samples=100,
#                     resources_per_trial={'gpu': 1},
#                     local_dir='/tmpscratch/alik/runlogs/ray_results',
#                     scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"))
