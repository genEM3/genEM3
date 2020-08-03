import os
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from genEM3.data.wkwdata import WkwData
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn, Decoder_4_sampling_bn
from genEM3.training.autoencoder import Trainer

# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, 'datasources.json')
input_shape = (302, 302, 1)
output_shape = (302, 302, 1)
data_sources = WkwData.datasources_from_json(datasources_json_path)

# With Caching (cache filled)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources
)

stats = dataset.get_datasource_stats(1)
print(stats)


