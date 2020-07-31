import os
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import device as torchDevice
from genEM3.util import gpu, viewData


class Predictor:

    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 state_dict: Optional[dict] = None,
                 device: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 output_shape: Optional[Tuple[int, ...]] = None):

        self.dataloader = dataloader
        self.model = model
        self.state_dict = state_dict
        self.device = device
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.out_shape = output_shape

    @torch.no_grad()
    def predict(self):
        """Predict using a dictionary of batches using the key: 'input'"""
        outputs = []
        for i, data in enumerate(self.dataloader):
            inputs = data['input']
            outputs.append(self.model(inputs))
            # viewData.data2fig_subplot(inputs, outputs, 1)
        return outputs

    @torch.no_grad()
    def predictList(self):
        """Predict when input is a list for each Batch"""
        outputs = []
        for i, data in enumerate(self.dataloader):
            inputs = data[0]
            outputs.append(self.model(inputs))
            # viewData.data2fig_subplot(inputs, outputs, 1)
        return outputs

    @torch.no_grad()
    def encode(self):
        """encode data into hidden representation and return the result as a list"""
        outputs = []
        for i, data in enumerate(self.dataloader):
            inputs = data['input']
            outputs.append(self.model.encode_input(inputs))
        return outputs

    @torch.no_grad()
    def encodeList(self):
        """encode list data"""
        outputs = []
        for i, data in enumerate(self.dataloader):
            inputs = data[0]
            outputs.append(self.model.encode_input(inputs))
        return outputs
