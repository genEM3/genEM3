from typing import Tuple, Sequence, Dict
from collections import namedtuple

import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import wkw

DataSource = namedtuple('DataSource',
    [
    'input_path',
    'input_dtype',
    'input_bbox',
    'label_path',
    'label_dtype',
    'label_bbox'
    ])


class Wkwdata(Dataset):

    def __init__(self,
                 data_sources: Sequence[DataSource],
                 data_strata: Dict,
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 ):

        self.data_sources = data_sources
        self.data_strata = data_strata
        self.input_shape = input_shape
        self.output_shape = output_shape

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
        print(corner_min_target, corner_max_target)
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
        source_idx = int(np.argmax(np.asarray(self.cube_idx_ranges_max) >= sample_idx))
        cube_key = list(self.data_object.keys())[source_idx]
        # Get appropriate subscript index for the respective training data cube, given the global linear index
        cube_sub = np.unravel_index(sample_idx - self.source_idx_ranges_min[source_idx],
                                    dims=self.cube_mesh_grids[source_idx]['target']['x'].shape)

        # Get target sample
        origin_target = np.asarray([
            self.cube_mesh_grids[source_idx]['target']['x'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[source_idx]['target']['y'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[source_idx]['target']['z'][cube_sub[0], cube_sub[1], cube_sub[2]],
        ]).astype(np.uint16)
        # ds_target = self.data_object[cube_key]['target']
        # target = D3d.crop_c(ds_target, origin_target, self.output_shape)

        # Get input sample
        origin_input = np.asarray([
            self.cube_mesh_grids[source_idx]['input']['x'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[source_idx]['input']['y'][cube_sub[0], cube_sub[1], cube_sub[2]],
            self.cube_mesh_grids[source_idx]['input']['z'][cube_sub[0], cube_sub[1], cube_sub[2]],
        ]).astype(np.uint16)
        # ds_input = self.data_object[cube_key]['input']
        # input_ = D3d.crop_c(ds_input, origin_input, self.input_shape)

        # if self.pad_target is True:
        #     target = self.pad(target)
        #
        # input_ = torch.from_numpy(np.reshape(input_, (1, input_.shape[0], input_.shape[1], input_.shape[2])))
        # target = torch.from_numpy(np.reshape(target, (1, target.shape[0], target.shape[1], target.shape[2])))

        # return input_, target

        pass

    def get_random_sample(self):

        """ Retrieves a random pair of input and target tensors from all available training cubes"""

        idx = random.sample(range(self.cube_idx_ranges_max[-1]), 1)
        input_, target = self.get_ordered_sample(idx)

        return input_, target

    def pad(self, target):
        pad_shape = np.floor((np.asarray(self.input_shape) - np.asarray(self.output_shape)) / 2).astype(int)
        target = np.pad(target,
                        ((pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1]), (pad_shape[2], pad_shape[2])),
                        'constant')

        return target


    @staticmethod
    def datasources_from_json(json_path):
        pass

    @staticmethod
    def datasources_to_json(json_path):
        pass


