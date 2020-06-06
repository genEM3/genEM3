import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn, Decoder_4_sampling_bn
from genEM3.training.training import Trainer


# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, 'datasources.json')
input_shape = (302, 302, 1)
output_shape = (302, 302, 1)
data_sources = WkwData.datasources_from_json(datasources_json_path)
# data_split = DataSplit(train=[1, 3], validation=[2, 4], test=[5])
data_split = DataSplit(train=0.7, validation=0.2, test=0.1)

dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split
)

train_sampler = SubsetRandomSampler(dataset.data_train_inds)
validation_sampler = SubsetRandomSampler(dataset.data_validation_inds)

train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, sampler=validation_sampler)

input_size = 302
output_size = input_size
valid_size = 17
kernel_size = 3
stride = 1
n_fmaps = 8
n_latent = 5000
model = AE(
    Encoder_4_sampling_bn(input_size, kernel_size, stride, n_fmaps, n_latent),
    Decoder_4_sampling_bn(output_size, kernel_size, stride, n_fmaps, n_latent))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.8)

num_epoch = 10
log_int = 10
device = 'cpu'

trainer = Trainer(run_root,
                  model,
                  optimizer,
                  criterion,
                  train_loader,
                  validation_loader,
                  num_epoch,
                  log_int,
                  device)

trainer.train()

