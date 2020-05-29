import os
import random
import json
from shutil import rmtree
from typing import Tuple, Sequence, Dict
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset
import wkw


DataSource = namedtuple(
    'DataSource',
    ['id',
     'input_path',
     'input_dtype',
     'input_bbox',
     'target_path',
     'target_dtype',
     'target_bbox'])


class WkwData(Dataset):

    def __init__(self,
                 data_sources: Sequence[DataSource],
                 data_strata: Dict,
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 pad_target: bool = False,
                 cache_root: str = None,
                 cache_wipe: bool = False,
                 cache_size: int = 1024,    # MiB
                 cache_dim: int = 0,
                 cache_range: int = 8   # times output_shape in cache_dim
                 ):

        if cache_root is not None:
            if not os.path.exists(cache_root):
                os.makedirs(cache_root)
            elif cache_wipe:
                rmtree(cache_root)

        self.data_sources = data_sources
        self.data_strata = data_strata
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pad_target = pad_target
        self.cache_root = cache_root
        self.cache_size = cache_size
        self.cache_dim = cache_dim
        self.cache_range = cache_range

        self.data_meshes = []
        self.data_inds_min = []
        self.data_inds_max = []

        self.get_data_meshes()
        self.get_data_ind_ranges()

    def __len__(self):
        return self.data_inds_max[-1]

    def __getitem__(self, idx):
        return self.get_ordered_sample(idx)

    def get_data_meshes(self):
        [self.data_meshes.append(self._get_data_mesh(i)) for i in range(len(self.data_sources))]

    def _get_data_mesh(self, data_source_idx):

        corner_min_target = np.ceil(np.asarray(self.data_sources[data_source_idx].input_bbox[0:3]) +
                                    np.asarray(self.input_shape) / 2 - np.asarray(self.output_shape) / 2).astype(int)
        n_fits = np.floor(np.asarray(self.data_sources[data_source_idx].input_bbox[3:6]) /
                                    np.asarray(self.output_shape)).astype(int)
        corner_max_target = corner_min_target + n_fits * np.asarray(self.output_shape)
        x = np.arange(corner_min_target[0], corner_max_target[0], self.output_shape[0])
        y = np.arange(corner_min_target[1], corner_max_target[1], self.output_shape[1])
        z = np.arange(corner_min_target[2], corner_max_target[2], self.output_shape[2])
        xm, ym, zm = np.meshgrid(x, y, z)
        mesh_target = {'x': xm, 'y': ym, 'z': zm}
        mesh_input = {'x': xm, 'y': ym, 'z': zm}

        return {'input': mesh_input, 'target': mesh_target}

    def get_data_ind_ranges(self):

        """ Computes the global linear idx limits contained in the respective training data cubes"""
        for source_idx, _ in enumerate(range(len(self.data_sources))):
            if source_idx == 0:
                self.data_inds_min.append(0)
            else:
                self.data_inds_min.append(self.data_inds_max[source_idx - 1] + 1)

            self.data_inds_max.append(self.data_inds_min[source_idx] +
                                      self.data_meshes[source_idx]['target']['x'].size - 1)

    def get_ordered_sample(self, sample_idx):

        """ Retrieves a pair of input and target tensors from all available training cubes based on the global linear
        sample_idx"""

        # Get appropriate training data cube sample_idx based on global linear sample_idx
        source_idx = int(np.argmax(np.asarray(self.data_inds_max) >= sample_idx))
        # Get appropriate subscript index for the respective training data cube, given the global linear index
        mesh_inds = np.unravel_index(sample_idx - self.data_inds_min[source_idx],
                                    dims=self.data_meshes[source_idx]['target']['x'].shape)

        # Get input sample
        origin_input = [
            int(self.data_meshes[source_idx]['input']['x'][mesh_inds[0], mesh_inds[1], mesh_inds[2]]),
            int(self.data_meshes[source_idx]['input']['y'][mesh_inds[0], mesh_inds[1], mesh_inds[2]]),
            int(self.data_meshes[source_idx]['input']['z'][mesh_inds[0], mesh_inds[1], mesh_inds[2]]),
        ]
        bbox_input = origin_input + list(self.input_shape)
        input_ = self.wkw_read_cached(self.data_sources[source_idx].input_path, bbox_input)

        # Get target sample
        origin_target = [
            self.data_meshes[source_idx]['target']['x'][mesh_inds[0], mesh_inds[1], mesh_inds[2]],
            self.data_meshes[source_idx]['target']['y'][mesh_inds[0], mesh_inds[1], mesh_inds[2]],
            self.data_meshes[source_idx]['target']['z'][mesh_inds[0], mesh_inds[1], mesh_inds[2]],
        ]
        bbox_target = origin_target + list(self.output_shape)
        target = self.wkw_read_cached(self.data_sources[source_idx].target_path, bbox_target)

        if self.pad_target is True:
            target = self.pad(target)

        input_ = torch.from_numpy(input_)
        target = torch.from_numpy(target)

        return input_, target

    def get_random_sample(self):

        """ Retrieves a random pair of input and target tensors from all available training cubes"""

        idx = random.sample(range(self.data_inds_max[-1]), 1)
        input_, target = self.get_ordered_sample(idx)

        return input_, target

    def pad(self, target):
        pad_shape = np.floor((np.asarray(self.input_shape) - np.asarray(self.output_shape)) / 2).astype(int)
        target = np.pad(target,
                        ((pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1]), (pad_shape[2], pad_shape[2])),
                        'constant')
        return target

    def wkw_read_cached(self, wkw_path, wkw_bbox):

        # If caching active
        if self.cache_root is not None:

            # Every 10th call check cache disk usage, if too large delete cache
            if (random.randint(0, 9) == 1) & (self.disk_usage(self.cache_root) > self.cache_size):
                rmtree('self.cache_root')

            # Generate elongated prefetching bbox in direction specified by self.cache_dim sized
            # self.cache_range*self.output_shape
            bbox_mult = np.zeros(3)
            bbox_mult[self.cache_dim] = self.cache_range
            wkw_cache_bbox = wkw_bbox[0:3] + \
                             list((np.asarray(wkw_bbox[3:6]) + np.asarray(self.output_shape) * bbox_mult).astype(int))
            wkw_cache_path = os.path.join(self.cache_root, wkw_path[1::])

            # If caching path does not exist create dirs and wkw dataset
            if not os.path.exists(wkw_cache_path):
                os.makedirs(wkw_cache_path)
                self.wkw_create(wkw_cache_path, self.wkw_header(wkw_path))

            # Attempt to load bbox from cache
            data = self.wkw_read(wkw_cache_path, wkw_bbox)

            # If data incomplete preload full cache bbox, write to cache and then load original bbox from cache
            if self.assert_data_completeness(data) is False:
                data = self.wkw_read(wkw_path, wkw_cache_bbox)
                self.wkw_write(wkw_cache_path, wkw_cache_bbox, data)
                data = self.wkw_read(wkw_cache_path, wkw_bbox)
        else:

            data = self.wkw_read(wkw_path, wkw_bbox)

        return data

    @staticmethod
    def wkw_header(wkw_path):
        with wkw.Dataset.open(wkw_path) as w:
            header = w.header

        return header

    @staticmethod
    def wkw_read(wkw_path, wkw_bbox):
        with wkw.Dataset.open(wkw_path) as w:
            data = w.read(wkw_bbox[0:3], wkw_bbox[3:6])

        return data

    @staticmethod
    def wkw_write(wkw_path, wkw_bbox, data):
        with wkw.Dataset.open(wkw_path) as w:
            w.write(wkw_bbox[0:3], data)

    @staticmethod
    def wkw_create(wkw_path, wkw_header):
        wkw.Dataset.create(wkw_path, wkw_header)

    @staticmethod
    def assert_data_completeness(data):
        if (np.any(data[:, 0, :, :]) & np.any(data[:, -1, :, :]) & np.any(data[:, :, 0, :]) & np.any(data[:, :, -1, :])
                & np.any(data[:, :, :, 0]) & np.any(data[:, :, :, -1])):
            flag = True
        else:
            flag = False
        return flag

    @staticmethod
    def disk_usage(root):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return np.floor(total_size/1024/1024) #MiB

    @staticmethod
    def datasources_from_json(json_path):
        with open(json_path) as f:
            datasources_dict = json.load(f)

        datasources = []
        for key in datasources_dict.keys():
            datasource = DataSource(**datasources_dict[key])
            datasources.append(datasource)

        return datasources

    @staticmethod
    def datasources_to_json(datasources, json_path):

        dumps = '{'
        for datasource_idx, datasource in enumerate(datasources):
            dumps += '\n    "datasource_{}"'.format(datasource.id) + ': {'
            dumps += json.dumps(datasource._asdict(), indent=8)[1:-1]
            dumps += "    },"
        dumps = dumps[:-1] + "\n}"

        with open(json_path, 'w') as f:
            f.write(dumps)






