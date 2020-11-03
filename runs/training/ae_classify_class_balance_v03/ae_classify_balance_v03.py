import os
import torch
import numpy as np

from genEM3.data import transforms
from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, AE_Encoder_Classifier, Classifier3Layered
from genEM3.training.classifier import Trainer, subsetWeightedSampler
from genEM3.util.path import getDataDir, gethostnameTimeString
# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
cache_HDD_root = os.path.join(getDataDir(), '.cache/')
datasources_json_path = os.path.join(getDataDir(), 'dense_3X_10_10_2_um/test_data_three_bboxes_with_myelin_v01.json')
state_dict_path = '/u/flod/code/genEM3/runs/training/ae_v05_skip/.log/epoch_60/model_state_dict'
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)

data_split = DataSplit(train=0.85, validation=0.15, test=0.00)
cache_RAM = True
cache_HDD = True
batch_size = 256
num_workers = 8

data_sources = WkwData.datasources_from_json(datasources_json_path)

transforms = transforms.Compose([
    transforms.RandomFlip(p=0.5, flip_plane=(1, 2)),
    transforms.RandomFlip(p=0.5, flip_plane=(2, 1)),
    transforms.RandomRotation90(p=1.0, mult_90=[0, 1, 2, 3], rot_plane=(1, 2))
])

dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split,
    transforms=transforms,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD,
    cache_HDD_root=cache_HDD_root
)
# Create the weighted samplers which create imbalance given the factor
# The sampler is linear between the given the clean sample imbalabce factor ranges
num_epoch = 1000
# controls the interval at which the dataloader's imbalance gets updated
loader_interval = 25
# The range of the imbalance (frequency ratio clean/debris)
imbalance_factor_range = [1, 19]
balance_factor_epoch = np.linspace(imbalance_factor_range[0], imbalance_factor_range[1], num=int(num_epoch/loader_interval))
# list of data loaders each contains a dictionary for train and validation loaders
data_loaders = [subsetWeightedSampler.get_data_loaders(dataset,
                                                       imbalance_factor=imbalance_factor,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers) for imbalance_factor in balance_factor_epoch]

input_size = 140
output_size = input_size
valid_size = 2
kernel_size = 3
stride = 1
n_fmaps = 16  # fixed in model class
n_latent = 2048
model = AE_Encoder_Classifier(
    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_latent=n_latent),
    Classifier3Layered(n_latent=n_latent, n_output=3))

checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_encoder_state_dict(state_dict)
model.freeze_encoder_weights(expr=r'^.*\.encoding_conv.*$')
model.reset_state()

for name, param in model.named_parameters():
    print(name, param.requires_grad)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00000075)

log_int = 5
device = 'cuda'
gpu_id = 1
save = True
save_int = 25
resume = False
run_name = f'class_balance_run_with_myelin_factor_{imbalance_factor_range[0]}_{imbalance_factor_range[1]}_{gethostnameTimeString()}'

trainer = Trainer(run_name=run_name,
                  run_root=run_root,
                  model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  data_loaders=data_loaders,
                  num_epoch=num_epoch,
                  log_int=log_int,
                  device=device,
                  save=save,
                  save_int=save_int,
                  resume=resume,
                  gpu_id=gpu_id)
trainer.train()
