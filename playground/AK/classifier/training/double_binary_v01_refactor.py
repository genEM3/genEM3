import os

import torch
import matplotlib

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, AE_Encoder_Classifier, Classifier3LayeredNoLogSoftmax
from genEM3.training.multiclass import Trainer
from genEM3.data.sampling import data_loaders_split
from genEM3.util.path import get_data_dir, gethostnameTimeString

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

# Data settings
run_root = os.path.dirname(os.path.abspath(__file__))
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)
data_split = DataSplit(train=0.70, validation=0.15, test=0.15)
cache_RAM = False
cache_HDD = False
batch_size = 1024
num_workers = 0

# Data sources
json_name = os.path.join(get_data_dir(), 'combined', 'combined_20K_patches.json')
data_sources = WkwData.read_short_ds_json(json_path=json_name)
transformations = WkwData.get_common_transforms()
# Data set
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split,
    transforms=transformations,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD)
# Data loaders
data_loader_params = {'dataset': dataset, 'batch_size': batch_size,
                      'num_workers': num_workers, 'collate_fn': dataset.collate_fn}
data_loaders = data_loaders_split(params=data_loader_params)
# Model initialization
input_size = 140
output_size = input_size
valid_size = 2
kernel_size = 3
stride = 1
n_fmaps = 16
n_latent = 2048
num_targets = 1
target_names = Trainer.target_strings(num_targets=num_targets)
model = AE_Encoder_Classifier(
    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_latent=n_latent),
    Classifier3LayeredNoLogSoftmax(n_latent=n_latent, n_output=num_targets))

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

criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.00000075)
num_epoch = 3000
log_int = 2
device = 'cuda'
gpu_id = 0
save = True
save_int = 25
resume_epoch = None
run_name = f'class_balance_run_with_myelin_factor_{gethostnameTimeString()}'
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
                  resume_epoch=resume_epoch,
                  gpu_id=gpu_id,
                  target_names=target_names)
trainer.train()
