import os

import torch
import numpy as np

from genEM3.data import transforms
from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, AE_Encoder_Classifier, Classifier3Layered
from genEM3.training.multiclass import Trainer, subsetWeightedSampler
from genEM3.util.path import getDataDir, gethostnameTimeString

## Train dataset: Create the dataset for training data
run_root = os.path.dirname(os.path.abspath(__file__))
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)

data_split = DataSplit(train=0.85, validation=0.15, test=0.00)
cache_RAM = True
cache_HDD = False
batch_size = 256
num_workers = 8
datasources_json_path = os.path.join(getDataDir(), 'dense_3X_10_10_2_um/original_merged_with_myelin_v01.json')
data_sources = WkwData.datasources_from_json(datasources_json_path)

transforms = transforms.Compose([
    transforms.RandomFlip(p=0.5, flip_plane=(1, 2)),
    transforms.RandomFlip(p=0.5, flip_plane=(2, 1)),
    transforms.RandomRotation90(p=1.0, mult_90=[0, 1, 2, 3], rot_plane=(1, 2))
])

train_dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split,
    transforms=transforms,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD)

## Test dataset: 10 bboxes of size 9 x 9 x 1 um:
# test dataset  
test_json_path = os.path.join(getDataDir(), '10x_test_bboxes/10X_9_9_1_um_with_myelin_v01.json')  
test_sources = WkwData.datasources_from_json(test_json_path)    
test_dataset = WkwData( 
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=test_sources,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD)
# Data Loaders:
# Create the weighted samplers which create imbalance given the factor
# The sampler is linear between the given the clean sample imbalabce factor ranges
num_epoch = 1000
# controls the interval at which the dataloader's imbalance gets updated
loader_interval = 50
# The range of the imbalance (frequency ratio clean/debris)
weight_range = [1, 1]
weight_range_epoch = np.linspace(weight_range[0], weight_range[1], num=int(num_epoch/loader_interval))
class_info = (('Non_myelin', 0, 1.0), ('Debris', 1, 1.0) , ('Myelin', 2, 1.0))
debris_index = [c[1] for c in class_info if c[0]=='Debris']
assert len(debris_index) == 1 
# list of data loaders each contains a dictionary for train and validation loaders
data_loaders = []
for w in weight_range_epoch:
    cur_weight_balance = [c[2] for c in class_info] 
    cur_weight_balance[debris_index[0]] = w 
    data_loaders.append(
                        subsetWeightedSampler.get_data_loaders(train_dataset,
                                                               imbalance_factor=cur_weight_balance,
                                                               test_dataset=test_dataset,
                                                               batch_size=batch_size,
                                                               num_workers=num_workers)
                        )
# Model initialization
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

# Load the encoder from the AE and freeze most weights
state_dict_path = '/u/flod/code/genEM3/runs/training/ae_v05_skip/.log/epoch_60/model_state_dict'
checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_encoder_state_dict(state_dict)
model.freeze_encoder_weights(expr=r'^.*\.encoding_conv.*$')
model.reset_state()
# print gradient requirement of parameters
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
run_name = f'class_balance_run_with_myelin_factor_{weight_range[0]}_{weight_range[1]}_{gethostnameTimeString()}'
class_target_value = [(c[1], c[0]) for c in class_info]
# Training Loop
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
                  gpu_id=gpu_id,
                  class_target_value=class_target_value)
trainer.train()
