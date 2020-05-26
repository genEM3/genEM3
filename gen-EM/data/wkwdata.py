import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Sequence
from collections import namedtuple
import wkw

DataSource = namedtuple('DataSource',
    [
    'idx',
    'input_path',
    'input_dtype',
    'input_bbox',
    'label_path',
    'label_dtype',
    'label_bbox'
    ])


class WkwDataset(Dataset):

    def __init__(self,
                 data_sources: Sequence[DataSource],
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 ):

        self.data_sources = data_sources
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.data_meshes = []

        self._get_data_meshes()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _get_data_meshes(self):
        a = 1


    @staticmethod
    def datasources_from_json(json_path):
        pass

    @staticmethod
    def datasources_to_json(json_path):
        pass


