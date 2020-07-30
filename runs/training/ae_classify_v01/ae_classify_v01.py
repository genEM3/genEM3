import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_1px_deep_convonly_skip, AE_Encoder_Classifier, Classifier
from genEM3.training.training import TrainerAE

# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
cache_HDD_root = os.path.join(run_root, '.cache/')
datasources_json_path = os.path.join(run_root, 'datasources_classifier_v01.json')
state_dict_path = os.path.join(run_root, 'ae_v05_skip_weights/epoch_36//model_state_dict')
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)

data_split = DataSplit(train=0.9, validation=0.1, test=0.0)
cache_RAM = True
cache_HDD = True
cache_root = os.path.join(run_root, '.cache/')
batch_size = 32
num_workers = 0

data_sources = WkwData.datasources_from_json(datasources_json_path)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=data_sources,
    data_split=data_split,
    cache_RAM=cache_RAM,
    cache_HDD=cache_HDD,
    cache_HDD_root=cache_root
)

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
n_fmaps = 16  # fixed in model class
n_latent = 2048
model = AE_Encoder_Classifier(
    Encoder_4_sampling_1px_deep_convonly_skip(input_size, kernel_size, stride, n_latent=n_latent),
    Classifier(n_latent=n_latent))

checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_encoder_state_dict(state_dict)
model.freeze_encoder_weights()
model.reset_state()

for name, param in model.named_parameters():
    print(name, param.requires_grad)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)

num_epoch = 10
log_int = 1
device = 'cpu'
save = True
resume = False

trainer = TrainerAE(run_root=run_root,
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




