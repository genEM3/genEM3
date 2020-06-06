import os
import json
import random
from typing import Tuple, Sequence, Union
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

DataStrata = namedtuple(
    'DataStrata',
    ['train_ids',
     'validation_ids',
     'test_ids']
)

DataSplit = namedtuple(
    'DataStrata',
    ['train_frac',
     'validation_frac',
     'test_frac']
)


class WkwData(Dataset):
    """Implements (2D/3D) pytorch Dataset subclass for wkw data"""

    def __init__(self,
                 data_sources: Sequence[DataSource],
                 data_strata: DataStrata,
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 norm_mean: float,
                 norm_std: float,
                 pad_target: bool = False,
                 cache_RAM: bool = True,
                 cache_HDD: bool = False,
                 cache_HDD_root: str = None
                 ):

        """
                Args:
                    data_sources:
                        Sequence of `DataSource` named tuples defining a given wkw data source and bounding box. Can
                        either be defined directly or generated from a datasource json file using the static
                        `WkwData.datasources_from_json` method.
                        Example (direct):
                            data_sources = [
                                wkwdata.Datasource(id=1, input_path='/path/to/wkw/input1', input_dtype='uint8',
                                    input_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z],
                                    target_path='/path/to/wkw/target1'), target_dtype='uint32',
                                    target_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z]),
                                wkwdata.Datasource(id=2, input_path='/path/to/wkw/input2', input_dtype='uint8',
                                    input_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z],
                                    target_path='/path/to/wkw/target2'), target_dtype='uint32',
                                    target_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z]),
                                ]
                        Example (json import):
                            data_sources = WkwData.datasources_from_json(datasources_json_path)
                    data_strata:
                        Defines stratification of data into training, validation and test sets. Can either be an
                        explicit assignment of specific data source ids to strata defined in a `DataStrata` named tuple.
                        Alternatively, fractions can be passed for the strata, which are then filled by random
                        assignment of data subsets.
                        Example (explicit):
                            To use data source ids (1,3,4) as train-, ids (2,6) as validation- and id 5 as test set:
                            data_strata = wkwdata.DataStrata(train=(1,3,4), validation=(2,6), test=5)
                        Example (implicit)
                            To use a 70% of the data as train, 20% as validation and 10% as test set:
                            data_strata = wkwdata.DataStrata(train=0.7, validation=0.2, test=0.1)
                    input_shape:
                        Input shape
                    output_shape:
                        Output shape


                """

        self.data_sources = data_sources
        self.data_strata = data_strata
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.pad_target = pad_target
        self.cache_RAM = cache_RAM
        self.cache_HDD = cache_HDD
        self.cache_HDD_root = cache_HDD_root

        self.data_cache = dict()
        self.data_meshes = []
        self.data_inds_min = []
        self.data_inds_max = []

        self.get_data_meshes()
        self.get_data_ind_ranges()

        if cache_RAM | cache_HDD:
            self.fill_caches()

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
        input_ = self.normalize(self.wkw_read_cached(self.data_sources[source_idx].input_path, bbox_input))

        # Get target sample
        origin_target = [
            self.data_meshes[source_idx]['target']['x'][mesh_inds[0], mesh_inds[1], mesh_inds[2]],
            self.data_meshes[source_idx]['target']['y'][mesh_inds[0], mesh_inds[1], mesh_inds[2]],
            self.data_meshes[source_idx]['target']['z'][mesh_inds[0], mesh_inds[1], mesh_inds[2]],
        ]
        bbox_target = origin_target + list(self.output_shape)
        target = self.normalize(self.wkw_read_cached(self.data_sources[source_idx].target_path, bbox_target))

        if self.pad_target is True:
            target = self.pad(target)

        input_ = torch.from_numpy(input_).float()
        if self.input_shape[2] == 1:
            input_ = input_.squeeze(3)

        target = torch.from_numpy(target).float()
        if self.output_shape[2] == 1:
            target = target.squeeze(3)

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

    def normalize(self, data):
        return (np.asarray(data)-self.norm_mean)/self.norm_std

    def fill_caches(self):
        for data_source_idx, data_source in enumerate(self.data_sources):
            print('Filling caches ... data source {}/{} input'.format(data_source_idx+1, len(self.data_sources)))
            self.fill_cache(data_source.input_path, data_source.input_bbox)
            print('Filling caches ... data source {}/{} target'.format(data_source_idx+1, len(self.data_sources)))
            self.fill_cache(data_source.target_path, data_source.target_bbox)

    def fill_cache(self, wkw_path, wkw_bbox):

        wkw_cache_path = os.path.join(self.cache_HDD_root, wkw_path[1::])

        # Attempt to read from HDD cache if already exists:
        if os.path.exists(os.path.join(wkw_cache_path, 'header.wkw')):
            data = self.wkw_read(wkw_cache_path, wkw_bbox)
            # If data incomplete read again from remote source
            if self.assert_data_completeness(data) is False:
                data = self.wkw_read(wkw_path, wkw_bbox)
        else:
            data = self.wkw_read(wkw_path, wkw_bbox)

        # If cache to RAM is true, save to RAM
        if self.cache_RAM:
            if wkw_path not in self.data_cache:
                self.data_cache[wkw_path] = {'data': data, 'wkw_bbox': wkw_bbox}

        # If cache to HDD is true, save to HDD
        if self.cache_HDD:
            if not os.path.exists(wkw_cache_path):
                os.makedirs(wkw_cache_path)

            if not os.path.exists(os.path.join(wkw_cache_path, 'header.wkw')):
                self.wkw_create(wkw_cache_path, self.wkw_header(wkw_path))

            self.wkw_write(wkw_cache_path, wkw_bbox, data)

    def wkw_read_cached(self, wkw_path, wkw_bbox):

        # Attempt to load bbox from RAM cache
        if wkw_path in self.data_cache:
            rel_pos = np.asarray(wkw_bbox[0:3]) - np.asarray(self.data_cache[wkw_path]['wkw_bbox'][0:3])
            data = self.data_cache[wkw_path]['data'][
                :,
                rel_pos[0]:rel_pos[0] + wkw_bbox[3],
                rel_pos[1]:rel_pos[1] + wkw_bbox[4],
                rel_pos[2]:rel_pos[2] + wkw_bbox[5],
                ]

        # Attempt to load bbox from HDD cache
        else:
            wkw_cache_path = os.path.join(self.cache_HDD_root, wkw_path[1::])
            data = self.wkw_read(wkw_cache_path, wkw_bbox)

        # If data incomplete, load conventionally
        if self.assert_data_completeness(data) is False:
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






