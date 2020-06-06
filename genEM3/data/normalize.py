import numpy as np
from torch.utils.data import Dataset

class Normalizer:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def get_stats_from_dataset(self, dataset: Dataset, num_samples=100):

        a = 1


