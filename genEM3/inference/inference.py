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
                 interpolate: str = None):

        self.model = model
        self.dataloader = dataloader
        self.datawriters = datawriters

        if output_prob_fn is None:
            output_prob_fn = lambda x: np.exp(x[:, 1, 0, 0])
        self.output_prob_fn = output_prob_fn
        self.interpolate = interpolate


    @torch.no_grad()
    def predict(self):
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Starting Inference ... ')
        start_time = time.time()
        sample_ind_phase = []
        for batch_idx, data in enumerate(self.dataloader):
            print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Predicting batch {}/{} ... '
                  .format(batch_idx, len(self.dataloader)))

            inputs = data['input']
            outputs = self.model(inputs)
            outputs_prob = np.round(self.output_prob_fn(outputs), 3)
            sample_ind_batch = data['sample_idx']
            sample_ind_phase.extend(sample_ind_batch)

            for datawriter in self.datawriters.values():
                datawriter.batch_to_cache(outputs, sample_ind_batch)

        elapsed_time = time.time() - start_time
        print(elapsed_time)

        for datawriter in self.datawriters.values():
            if self.interpolate is not None:
                datawriter.interpolate_sparse_cache(method=self.interpolate)

            datawriter.cache_to_wkw(output_wkw_root=datawriter.output_path)

