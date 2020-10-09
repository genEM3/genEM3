import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data import transforms
from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, AE_Encoder_Classifier, Classifier3Layered
from genEM3.training.classifier import Trainer

# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
run_name = 'run_w_pr'
cache_HDD_root = os.path.join(run_root, '../../../data/.cache/')
datasources_json_path = os.path.join(run_root, '../../../data/debris_clean_added_bboxes2_wiggle_datasource.json')
state_dict_path = os.path.join(run_root, '../ae_v05_skip/.log/epoch_60/model_state_dict')
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)

data_split = DataSplit(train=0.85, validation=0.15, test=0.00)
cache_RAM = False
cache_HDD = False
cache_root = os.path.join(run_root, '.cache/')
batch_size = 256
num_workers = 12

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

train_sampler = SubsetRandomSampler(dataset.data_train_inds)
train_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
    collate_fn=dataset.collate_fn)

validation_sampler = SubsetRandomSampler(dataset.data_validation_inds)
validation_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=validation_sampler,
    collate_fn=dataset.collate_fn)

data_loaders = {
    "train": train_loader,
    "val": validation_loader}

input_size = 140
output_size = input_size
valid_size = 2
kernel_size = 3
stride = 1
n_fmaps = 16  # fixed in model class
n_latent = 2048
model = AE_Encoder_Classifier(
    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_latent=n_latent),
    Classifier3Layered(n_latent=n_latent))

checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_encoder_state_dict(state_dict)
model.freeze_encoder_weights(expr=r'^.*\.encoding_conv.*$')
model.reset_state()

for name, param in model.named_parameters():
    print(name, param.requires_grad)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00000075)

num_epoch = 700
log_int = 5
device = 'cuda'
save = True
save_int = 25
resume = False

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
                  resume=resume)

trainer.train()




