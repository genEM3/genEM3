import os
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from genEM3.data import transforms
from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, AE_Encoder_Classifier, Classifier3Layered
from genEM3.training.classifier import Trainer

import numpy as np
# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))
run_name = 'class_balance_run_w_pr'
cache_HDD_root = os.path.join(run_root, '../../../data/.cache/')
datasources_json_path = os.path.join(run_root, '../../../data/debris_clean_added_bboxes2_wiggle_datasource.json')
state_dict_path = '/u/flod/code/genEM3/runs/training/ae_v05_skip/.log/epoch_60/model_state_dict'
input_shape = (140, 140, 1)
output_shape = (140, 140, 1)

data_split = DataSplit(train=0.85, validation=0.15, test=0.00)
cache_RAM = True
cache_HDD = True
cache_root = os.path.join(run_root, '.cache/')
batch_size = 256
num_workers = 0

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
########
# Get the target (debris vs. clean) for each sample
total_sample_range = iter(dataset.data_train_inds)
total_sample_set = set(dataset.data_train_inds)
# check uniqueness of the train indices
assert len(total_sample_set) == len(dataset.data_train_inds)
target_class = np.asarray([dataset.get_target_from_sample_idx(sample_idx) for sample_idx in total_sample_range], dtype=np.int32)
# print the sample imbalance
print('Target balance for original train set clean/debris: {}/{}'.format(
    len(np.where(target_class == 0)[0]), len(np.where(target_class == 1)[0])))
# Use the inverse of the number of samples as weight to create balance
class_sample_count = np.array(
    [len(np.where(target_class == t)[0]) for t in np.unique(target_class)])
# imbalance factor clean
for factor in range(1, 20):
    imbalance_factor_clean = factor 
    weight = 1. / class_sample_count
    weight[0] = weight[0]*imbalance_factor_clean
    samples_weight = np.array([weight[t] for t in target_class])
    # Create the weighted sampler
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    # Subset dataset 
    train_dataset = Subset(dataset, dataset.data_train_inds)
    subset_weighted_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
        collate_fn=dataset.collate_fn)
    print(f'########\nImbalance Factor: {imbalance_factor_clean}') 
    ratio_clean = list()
    for i, data in enumerate(subset_weighted_loader):
        print(f'####\nBatch Index: {i}/{len(subset_weighted_loader)}')
        # check that all sample indices are part of the total indices
        batch_idx = data['sample_idx']
        batch_idx_set = set(batch_idx)
        assert batch_idx_set.issubset(total_sample_set)
        repetition_num = len(batch_idx)-len(batch_idx_set)
        print(f'Repeated/total number of samples in current batch: {repetition_num}/{len(batch_idx)}')
        y = data['target']
        clean_num = float((y == 0).sum())
        debris_num = float((y == 1).sum())
        fraction_clean = clean_num / (debris_num + clean_num) 
        ratio_clean.append(clean_num / debris_num)
        print(f"Number of clean/debris samples in mini-batch: {int(clean_num)}/{int(debris_num)}\nFraction clean: {fraction_clean:.2f}, Ratio clean: {ratio_clean[i]:.2f}")
    # Show an example from each batch (clean and debris)
    example_idx = {'clean':np.where(y == 0)[0][0], 'debris': np.where(y == 1)[0][0]}
    for indexInBatch in example_idx.values():
        dataset.show_sample(batch_idx[indexInBatch])
    average_ratio_clean = np.asarray(ratio_clean).mean()
    print(f'The empirical average imbalanced factor: {average_ratio_clean:.2f}')

#########
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
device = 'cpu'
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
