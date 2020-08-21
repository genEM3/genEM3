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
from genEM3.inference.writer import DataWriter

class Predictor:

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 datawriters: Sequence[DataWriter],
                 output_prob_fn: Callable = None,
                 device: str = 'cpu',
                 gpu_id: int = None,
                 interpolate: str = None):

        self.model = model
        self.dataloader = dataloader
        self.datawriters = datawriters

        if output_prob_fn is None:
            output_prob_fn = lambda x: np.exp(x[:, 1, 0, 0])
        self.output_prob_fn = output_prob_fn
        self.interpolate = interpolate

        if device == 'cuda':
            gpu.get_gpu(gpu_id)
            device = torch.device(torch.cuda.current_device())

        self.device = torchDevice(device)


    @torch.no_grad()
    def predict(self):
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Starting Inference ... ')
        start_time = time.time()
        sample_ind_phase = []

        for batch_idx, data in enumerate(self.dataloader):

            try:

                print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Predicting batch {}/{} ... '
                      .format(batch_idx, len(self.dataloader)))

                inputs = data['input'].to(self.device)
                outputs = self.model(inputs)
                targets = data['target'].to(self.device)

                inputs, outputs, targets = Predictor.copy2cpu(inputs, outputs, targets)

                outputs_prob = np.round(self.output_prob_fn(outputs), 3)
                sample_ind_batch = data['sample_idx']
                sample_ind_phase.extend(sample_ind_batch)

                for datawriter in self.datawriters.values():
                    datawriter.batch_to_cache(outputs, sample_ind_batch)

            except:
                print('error at batch {}'.format(batch_idx))


        elapsed_time = time.time() - start_time
        print(elapsed_time)

        for datawriter in self.datawriters.values():
            if self.interpolate is not None:
                datawriter.interpolate_sparse_cache(method=self.interpolate)

            datawriter.cache_to_wkw(output_wkw_root=datawriter.output_path)

    @staticmethod
    def copy2cpu(inputs, outputs, targets):
        if inputs.is_cuda:
            inputs = inputs.cpu()
        if outputs.is_cuda:
            outputs = outputs.cpu()
        if targets.is_cuda:
            targets = targets.cpu()
        return inputs, outputs, targets