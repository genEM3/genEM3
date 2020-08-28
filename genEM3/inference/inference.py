import os
import time
import datetime
from typing import Sequence, Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import device as torchDevice
from genEM3.util import gpu, viewData

class Predictor:

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 output_prob_fn: Callable = None,
                 output_dtype_fn: Callable = None,
                 output_dtype: np.dtype = None,
                 output_label: str = None,
                 output_wkw_root: str = None,
                 output_wkw_compress: bool = False,
                 device: str = 'cpu',
                 gpu_id: int = None,
                 interpolate: str = None):

        self.model = model
        self.dataloader = dataloader

        if output_prob_fn is None:
            output_prob_fn = lambda x: np.exp(x[:, 1, 0, 0])

        self.output_dtype_fn = output_dtype_fn
        self.output_dtype = output_dtype
        self.output_prob_fn = output_prob_fn
        self.output_label = output_label
        self.output_wkw_root = output_wkw_root

        if output_wkw_compress is False:
            self.output_wkw_block_type = 1
        else:
            self.output_wkw_block_type = 2


        if device == 'cuda':
            gpu.get_gpu(gpu_id)
            device = torch.device(torch.cuda.current_device())

        self.device = torchDevice(device)
        self.interpolate = interpolate


    @torch.no_grad()
    def predict(self):
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Starting Inference ... ')
        start_time = time.time()
        sample_ind_phase = []

        for batch_idx, data in enumerate(self.dataloader):

            print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Predicting batch {}/{} ... '
                  .format(batch_idx, len(self.dataloader)))

            sample_ind_batch = data['sample_idx']
            sample_ind_phase.extend(sample_ind_batch)
            inputs = data['input'].to(self.device)
            outputs = Predictor.copy2cpu(self.model(inputs))

            self.dataloader.dataset.write_output_to_cache(
                self.output_dtype_fn(self.output_prob_fn(outputs).data.numpy()).astype(self.output_dtype),
                sample_ind_batch, self.output_label)

        if self.interpolate is not None:
            self.dataloader.dataset.interpolate_sparse_cache(output_label=self.output_label, method=self.interpolate)

        if self.output_wkw_root is not None:
            self.dataloader.dataset.wkw_write_cache(
                output_label=self.output_label,
                output_wkw_root=self.output_wkw_root,
                output_dtype=self.output_dtype,
                output_block_type=self.output_wkw_block_type)

    @staticmethod
    def copy2cpu(data):
        if data.is_cuda:
            data = data.cpu()
        return data
