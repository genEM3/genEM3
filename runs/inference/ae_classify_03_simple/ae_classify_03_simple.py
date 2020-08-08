import os

import numpy as np
from scipy.special import logit, expit

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE_Encoder_Classifier, Encoder_4_sampling_bn_1px_deep_convonly_skip, Classifier
from genEM3.inference.inference import Predictor
from genEM3.inference.writer import DataWriter

run_root = os.path.dirname(os.path.abspath(__file__))
cache_HDD_root = os.path.join(run_root, '.cache/')
datasources_json_path = os.path.join(run_root, 'datasources_distributed_test.json')
state_dict_path = os.path.join(run_root, '../../training/ae_classify_v03_1layer_unfreeze_latent_debris_clean/.log/run_w_pr/epoch_30/model_state_dict')
device = 'cpu'

batch_size = 16
input_shape = (140, 140, 1)
output_shape = (1, 1, 1)
num_workers = 0

kernel_size = 3
stride = 1
n_fmaps = 16
n_latent = 2048
input_size = 140
output_size = input_size
model = AE_Encoder_Classifier(
    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_latent=n_latent),
    Classifier(n_latent=n_latent))

datasources = WkwData.datasources_from_json(datasources_json_path)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=datasources,
    stride=(140, 140, 1),
    cache_HDD=True,
    cache_RAM=True,
    cache_HDD_root=cache_HDD_root
)

prediction_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers)

checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

def prob_collate_fn(outputs):
    outputs_collate = list(np.exp(outputs.detach().numpy())[:, 0, :, :])
    return outputs_collate

output_dtype = np.uint8
output_dtype_fn = lambda x: (logit(x) + 6) * 256 / 12
output_dtype_fni = lambda x: expit(x * 12 / 256) - 6

datawriter_prob = DataWriter(
    dataloader=prediction_loader,
    output_collate_fn=prob_collate_fn,
    output_label='prediction_probabilities',
    output_path=run_root,
    output_dtype=output_dtype,
    output_dtype_fn=output_dtype_fn
)

data_writers = {'prediction_probabilities': datawriter_prob}

predictor = Predictor(
    dataloader=prediction_loader,
    datawriters=data_writers,
    model=model,
    state_dict=state_dict,
    device=device,
    batch_size=batch_size,
    input_shape=input_shape,
    output_shape=output_shape,
    interpolate='nearest')

predictor.predict()



