import os
import datetime
from typing import Optional, Tuple, Sequence
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
                 dataloader: DataLoader,
                 datawriters: Sequence[DataWriter],
                 model: torch.nn.Module,
                 state_dict: Optional[dict] = None,
                 device: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 output_shape: Optional[Tuple[int, ...]] = None,
                 interpolate: str = None):

        self.dataloader = dataloader
        self.datawriters = datawriters
        self.model = model
        self.state_dict = state_dict
        self.device = device
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.out_shape = output_shape
        self.interpolate = interpolate

    @torch.no_grad()
    def predict(self):
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Starting Inference ... ')

        for i, (data, index) in enumerate(self.dataloader):
            print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Predicting batch {}/{} ... '
                  .format(i, len(self.dataloader)))

            inputs = data['input']
            outputs = self.model(inputs)

            for datawriter in self.datawriters.values():
                datawriter.batch_to_cache(outputs, index)

        for datawriter in self.datawriters.values():
            if self.interpolate is not None:
                datawriter.interpolate_sparse_cache(method=self.interpolate)

            datawriter.cache_to_wkw(output_wkw_root=datawriter.output_path)


