import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE_Encoder_Classifier, Encoder_4_sampling_bn_1px_deep_convonly_skip, Classifier
from genEM3.util.latent_sampler import LatentSampler

run_root = os.path.dirname(os.path.abspath(__file__))
cache_HDD_root = os.path.join(run_root, '../../../data/.cache/')
datasources_json_path = os.path.join(run_root, '../../../data/debris_clean_datasource.json')
state_dict_path = os.path.join(run_root, '../../training/ae_classify_v03_1layer_unfreeze_latent_debris_clean/.log/run_w_pr/epoch_30/model_state_dict')
device = 'cpu'

data_split = DataSplit(train=1.0, validation=0.0, test=0.0)

batch_size = 5
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
    data_split=data_split,
    cache_HDD=True,
    cache_RAM=True,
    cache_HDD_root=cache_HDD_root
)

train_sampler = SubsetRandomSampler(dataset.data_train_inds)
train_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
    collate_fn=dataset.collate_fn)

checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

latentSampler = LatentSampler(run_root=run_root, model=model, dataloader=train_loader)
latentSampler.sample()
latentSampler.pca(n_components=3, plot=True)
a = 1



