import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn_1px_deep, Decoder_4_sampling_bn_1px_deep
from genEM3.training.training import Trainer


# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, 'datasources_distributed.json')
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)
data_sources = WkwData.datasources_from_json(datasources_json_path)
# data_split = DataSplit(train=[1], validation=[], test=[])
data_split = DataSplit(train=0.75, validation=0.15, test=0.1)
cache_RAM = True
cache_HDD = False
cache_root = os.path.join(run_root, '.cache/')
batch_size = 128
num_workers = 8

dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD,
    cache_HDD_root=cache_root
)

# dataset.update_datasources_stats()
# dataset.datasources_to_json(dataset.data_sources, os.path.join(run_root, 'datasources_auto_stat.json'))

train_sampler = SubsetRandomSampler(dataset.data_train_inds)
validation_sampler = SubsetRandomSampler(dataset.data_validation_inds)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
    collate_fn=dataset.collate_fn)
validation_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=validation_sampler,
    collate_fn=dataset.collate_fn)

input_size = 140
output_size = input_size
valid_size = 2
kernel_size = 3
stride = 1
n_fmaps = 16
n_latent = 192
model = AE(
    Encoder_4_sampling_bn_1px_deep(input_size, kernel_size, stride, n_fmaps, n_latent),
    Decoder_4_sampling_bn_1px_deep(output_size, kernel_size, stride, n_fmaps, n_latent))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.8)

num_epoch = 100
log_int = 100
device = 'cuda'
save = True
resume = False

trainer = Trainer(run_root=run_root,
                  model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  train_loader=train_loader,
                  validation_loader=validation_loader,
                  num_epoch=num_epoch,
                  log_int=log_int,
                  device=device,
                  save=save,
                  resume=resume)


trainer.train()

