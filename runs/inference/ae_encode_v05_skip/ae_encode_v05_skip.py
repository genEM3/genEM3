import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from genEM3.data.wkwdata import WkwData, DataSplit
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn_1px_deep_convonly_skip, Decoder_4_sampling_bn_1px_deep_convonly_skip
from genEM3.inference.inference import Predictor

run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, 'datasources_distributed.json')
state_dict_path = os.path.join(run_root, '../../training/ae_v05_skip/.log/torch_model')
device = 'cpu'

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
model = AE(
    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_fmaps, n_latent),
    Decoder_4_sampling_bn_1px_deep_convonly_skip(output_size, kernel_size, stride, n_fmaps, n_latent))

datasources = WkwData.datasources_from_json(datasources_json_path)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=output_shape,
    data_sources=datasources,
)

prediction_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, num_workers=num_workers)

checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
predictor = Predictor(
    dataloader=prediction_loader,
    model=model,
    state_dict=state_dict,
    device=device,
    batch_size=batch_size,
    input_shape=input_shape,
    output_shape=output_shape)
predictor.predict()



