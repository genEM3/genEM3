import os
import json
import random
import numpy as np
from collections import namedtuple
from typing import Tuple, Sequence, List, Callable
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import wkw

np.random.seed(1337)

DataSourceDefaults = (
    ("id", str),
    ("input_path", 'NaN'),
    ("input_bbox", 'NaN'),
    ("input_mean", 'NaN'),
    ("input_std", 'NaN'),
    ("target_path", 'NaN'),
    ("target_bbox", 'NaN'),
    ("target_class", 'NaN'),
    ("target_binary", 'NaN'),
)

DataSource = namedtuple(
    'DataSource',
    [fields[0] for fields in list(DataSourceDefaults)],
    defaults=[defaults[1] for defaults in list(DataSourceDefaults)])

DataSplit = namedtuple(
    'DataSplit',
    ['train',
     'validation',
     'test']
)


class WkwData(Dataset):
    """Implements (2D/3D) pytorch Dataset subclass for wkw data"""

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 target_shape: Tuple[int, int, int],
                 data_sources: Sequence[DataSource],
                 data_split: DataSplit = None,
                 stride: Tuple[int, int, int] = None,
                 normalize: bool = True,
                 transforms: Callable = None,
                 pad_target: bool = False,
                 cache_RAM: bool = True,
                 cache_HDD: bool = False,
                 cache_HDD_root: str = None
                 ):

        """
                Args:
                    input_shape:
                        Specifies (x,y,z) dimensions of input patches in pixels
                    target_shape:
                        Specifies (x,y,z) dimensions of target patches in pixels
                    data_sources:
                        Sequence of `DataSource` named tuples defining a given wkw data source and bounding box. Can
                        either be defined directly or generated from a datasource json file using the static
                        `WkwData.datasources_from_json` method.
                        Example (direct):
                            data_sources = [
                                wkwdata.Datasource(id=1, input_path='/path/to/wkw/input1',
                                    input_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z], input_mean=148.0,
                                    input_std=36.0, target_path='/path/to/wkw/target1'),
                                    target_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z]),
                                wkwdata.Datasource(id=2, input_path='/path/to/wkw/input2',
                                    input_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z], input_mean=148.0,
                                    input_std=36.0, target_path='/path/to/wkw/target2'),
                                    target_bbox=[pos_x, pos_y, pos_z, ext_x, ext_y, ext_z])]
                        Example (json import):
                            data_sources = WkwData.datasources_from_json(datasources_json_path)
                    data_split:
                        Defines how provided data is split into training, validation and test sets. The split can either
                        be defined as strata (define specific data sources to serve as train, val, test sets) or
                        as fractions (define subsets of samples drawn from all data sources to serve as train, val, test
                        sets).
                        Example (strata):
                            To use data source ids (1,3,4) as train-, ids (2,6) as validation- and id 5 as test set:
                            data_split = wkwdata.DataSplit(train=[1,3,4], validation=[2,6], test=[5])
                        Example (fractions)
                            To use a 70% of the data as train, 20% as validation and 10% as test set:
                            data_split = wkwdata.DataSplit(train=0.7, validation=0.2, test=0.1)
                    normalize:
                        If true, patches are normalized to standard normal using input mean and std specified in
                        the respective datasource
                    pad_target:
                        If true, target patches are padded to the same shape as input patches
                    cache_RAM:
                        If true, all data is cached into RAM for faster training.
                    cache_HDD:
                        If true, all data is cached to HDD for faster training (Mostly relevant if data is hosted at a
                        remote location and bandwidth to local instance is limited or multiple processes need to access
                        the same path).
                    cache_HDD_root:
                        Path on the local filesystem where HDD cache should be created

                """

        if cache_HDD_root is None:
            cache_HDD_root = '.'

        self.input_shape = input_shape
        self.output_shape = target_shape
        self.data_sources = data_sources
        self.data_split = data_split

        if stride is None:
            self.stride = target_shape
        else:
            self.stride = stride

        self.normalize = normalize
        self.transforms = transforms
        self.pad_target = pad_target
        self.cache_RAM = cache_RAM
        self.cache_HDD = cache_HDD

        self.cache_HDD_root = cache_HDD_root

        self.data_cache_input = dict()
        self.data_cache_output = dict()
        self.data_meshes = []
        self.data_inds_min = []
        self.data_inds_max = []

        self.data_train_inds = []
        self.data_validation_inds = []
        self.data_test_inds = []

        self.get_data_meshes()
        self.get_data_ind_ranges()

        if data_split is not None:
            self.get_data_ind_splits()

        if cache_RAM | cache_HDD:
            self.fill_caches()

    def __len__(self):
        return self.data_inds_max[-1]

    def __getitem__(self, idx):
        return self.get_ordered_sample(idx)

    def get_data_meshes(self):
        [self.data_meshes.append(self.get_data_mesh(i)) for i in range(len(self.data_sources))]

    def get_data_mesh(self, data_source_idx):

        corner_min_target = np.floor(np.asarray(self.data_sources[data_source_idx].input_bbox[0:3]) +
                                    np.asarray(self.input_shape) / 2).astype(int)
        n_fits = np.ceil((np.asarray(self.data_sources[data_source_idx].input_bbox[3:6]) -
                           np.asarray(self.input_shape) / 2) / np.asarray(self.stride)).astype(int)
        corner_max_target = corner_min_target + n_fits * np.asarray(self.stride)
        x = np.arange(corner_min_target[0], corner_max_target[0], self.stride[0])
        y = np.arange(corner_min_target[1], corner_max_target[1], self.stride[1])
        z = np.arange(corner_min_target[2], corner_max_target[2], self.stride[2])
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

    def get_data_ind_splits(self):

        if type(self.data_split.train) is float:
            # Create variable for the maximum index of training examples 
            maxIndex = self.data_inds_max[-1]
            data_inds_all = list(range(maxIndex+1))
            data_inds_all_rand = np.random.permutation(data_inds_all)
            train_idx_max = int(self.data_split.train*maxIndex)
            data_train_inds = list(data_inds_all_rand[0:train_idx_max])
            validation_idx_max = train_idx_max + int(self.data_split.validation * maxIndex)
            data_validation_inds = list(data_inds_all_rand[train_idx_max+1:validation_idx_max])
            test_idx_max = validation_idx_max + int(self.data_split.test * maxIndex)
            data_test_inds = list(data_inds_all_rand[validation_idx_max+1:test_idx_max])
        else:
            data_train_inds = []
            for i, id in enumerate(self.data_split.train):
                idx = self.datasource_id_to_idx(id)
                data_train_inds += list(range(self.data_inds_min[idx], self.data_inds_max[idx]+1))

            data_validation_inds = []
            for i, id in enumerate(self.data_split.validation):
                idx = self.datasource_id_to_idx(id)
                data_validation_inds += list(range(self.data_inds_min[idx], self.data_inds_max[idx] + 1))

            data_test_inds = []
            for i, id in enumerate(self.data_split.test):
                idx = self.datasource_id_to_idx(id)
                data_test_inds += list(range(self.data_inds_min[idx], self.data_inds_max[idx] + 1))

        self.data_train_inds = data_train_inds
        self.data_validation_inds = data_validation_inds
        self.data_test_inds = data_test_inds

    def get_ordered_sample(self, sample_idx, normalize=None):

        """ Retrieves a pair of input and target tensors from all available training cubes based on the global linear
        sample_idx"""

        if normalize is None:
            normalize = self.normalize

        source_idx, bbox_input = self.get_bbox_for_sample_idx(sample_idx, sample_type='input')
        if self.cache_RAM | self.cache_HDD:
            input_ = self.wkw_read_cached(source_idx, 'input', bbox_input)
        else:
            input_ = self.wkw_read(self.data_sources[source_idx].input_path, bbox_input)

        if normalize:
            input_ = WkwData.normalize(input_, self.data_sources[source_idx].input_mean,
                                    self.data_sources[source_idx].input_std)

        source_idx, bbox_target = self.get_bbox_for_sample_idx(sample_idx, sample_type='target')
        if self.data_sources[source_idx].target_binary == 1:
            target = np.asarray(self.data_sources[source_idx].target_class)
        else:
            if (self.data_sources[source_idx].input_path == self.data_sources[source_idx].target_path) & \
                    (bbox_input == bbox_target):
                target = input_
            else:
                if self.cache_RAM | self.cache_HDD:
                    target = self.wkw_read_cached(source_idx, 'target', bbox_target)
                else:
                    target = self.wkw_read(self.data_sources[source_idx].target_path, bbox_target)

        if self.pad_target is True:
            target = self.pad(target)

        input_ = torch.from_numpy(input_).float()
        if self.input_shape[2] == 1:
            input_ = input_.squeeze(3)

        if self.transforms:
            input_ = self.transforms(input_)

        if self.data_sources[source_idx].target_binary == 1:
            target = torch.from_numpy(target).long()
        else:
            target = torch.from_numpy(target).float()
            if self.output_shape[2] == 1:
                target = target.squeeze(3)
            if self.transforms:
                target = self.transforms(target)

        return {'input': input_, 'target': target, 'sample_idx': sample_idx}

    def write_output_to_cache(self,
                              outputs: List[np.ndarray],
                              sample_inds: List[int],
                              output_label: str):

        if type(sample_inds) is torch.Tensor:
            sample_inds = sample_inds.data.numpy().tolist()

        for output_idx, sample_idx in enumerate(sample_inds):
            source_idx, bbox = self.get_bbox_for_sample_idx(sample_idx, 'target')
            print(bbox)

            wkw_path = self.data_sources[source_idx].input_path
            wkw_bbox = self.data_sources[source_idx].input_bbox

            if wkw_path not in self.data_cache_output:
                self.data_cache_output[wkw_path] = {}

            if str(wkw_bbox) not in self.data_cache_output[wkw_path]:
                self.data_cache_output[wkw_path][str(wkw_bbox)] = {}

            if output_label not in self.data_cache_output[wkw_path][str(wkw_bbox)]:
                data = np.full(wkw_bbox[3:6], np.nan, dtype=np.float32)
                self.data_cache_output[wkw_path][str(wkw_bbox)][output_label] = data

            data_min = np.asarray(bbox[0:3]) - np.asarray(wkw_bbox[0:3])
            data_max = data_min + np.asarray(bbox[3:6])

            data = self.data_cache_output[wkw_path][str(wkw_bbox)][output_label]
            data[data_min[0]:data_max[0], data_min[1]:data_max[1], data_min[2]:data_max[2]] = outputs[output_idx].reshape(self.output_shape)
            self.data_cache_output[wkw_path][str(wkw_bbox)][output_label] = data


    def get_random_sample(self):
        """ Retrieves a random pair of input and target tensors from all available training cubes"""

        sample_idx = random.sample(range(self.data_inds_max[-1]), 1)

        return self.get_ordered_sample(sample_idx)

    def pad(self, target):
        pad_shape = np.floor((np.asarray(self.input_shape) - np.asarray(self.output_shape)) / 2).astype(int)
        target = np.pad(target,
                        ((pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1]), (pad_shape[2], pad_shape[2])),
                        'constant')
        return target

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
            if wkw_path not in self.data_cache_input:
                self.data_cache_input[wkw_path] = {str(wkw_bbox): data}
            else:
                self.data_cache_input[wkw_path][str(wkw_bbox)] = data

        # If cache to HDD is true, save to HDD
        if self.cache_HDD:
            if not os.path.exists(wkw_cache_path):
                os.makedirs(wkw_cache_path)

            if not os.path.exists(os.path.join(wkw_cache_path, 'header.wkw')):
                self.wkw_create(wkw_cache_path, self.wkw_header(wkw_path))

            self.wkw_write(wkw_cache_path, wkw_bbox, data)

    def wkw_read_cached(self, source_idx, source_type, wkw_bbox):

        key = source_type+'_path'
        key_idx = self.data_sources[source_idx]._fields.index(key)
        wkw_path = self.data_sources[source_idx][key_idx]

        key = source_type + '_bbox'
        key_idx = self.data_sources[source_idx]._fields.index(key)
        abs_pos = self.data_sources[source_idx][key_idx]

        # Attempt to load bbox from RAM cache
        if (wkw_path in self.data_cache_input) & (str(abs_pos) in self.data_cache_input[wkw_path]):

            rel_pos = np.asarray(wkw_bbox[0:3]) - np.asarray(abs_pos[0:3])
            data = self.data_cache_input[wkw_path][str(abs_pos)][
                :,
                rel_pos[0]:rel_pos[0] + wkw_bbox[3],
                rel_pos[1]:rel_pos[1] + wkw_bbox[4],
                rel_pos[2]:rel_pos[2] + wkw_bbox[5],
                ]

        # Attempt to load bbox from HDD cache
        else:
            wkw_cache_path = os.path.join(self.cache_HDD_root, wkw_path[1::])
            if os.path.exists(os.path.join(wkw_cache_path, 'header.wkw')):
                data = self.wkw_read(wkw_cache_path, wkw_bbox)
            # If data incomplete, load conventionally
            else:
                data = self.wkw_read(wkw_path, wkw_bbox)

        return data

    def wkw_write_cached(self,
                         wkw_path,
                         wkw_bbox,
                         output_wkw_root,
                         output_label,
                         output_dtype=None,
                         output_dtype_fn=None):

        if output_dtype is None:
            output_dtype = np.float

        if output_dtype_fn is None:
            output_dtype_fn = lambda x: x

        tmp, wkw_mag = os.path.split(wkw_path)
        data = np.expand_dims(output_dtype_fn(self.data_cache_output[wkw_path][wkw_bbox][output_label])
                              .astype(output_dtype), axis=0)

        output_wkw_path = os.path.join(output_wkw_root, output_label, wkw_mag)
        if not os.path.exists(output_wkw_path):
            os.makedirs(output_wkw_path)
            self.wkw_create(output_wkw_path)

        print('Writing cache to wkw ... ' + output_wkw_path + ' | ' + wkw_bbox)
        bbox_from_str = [int(x) for x in wkw_bbox[1:-1].split(',')]
        self.wkw_write(output_wkw_path, bbox_from_str, data)

    def get_bbox_for_sample_idx(self, sample_idx, sample_type='input'):
        source_idx, mesh_inds = self.get_source_mesh_for_sample_idx(sample_idx)
        if sample_type == 'input':
            shape = self.input_shape
        else:
            shape = self.output_shape
        origin = [
            int(self.data_meshes[source_idx][sample_type]['x'][mesh_inds[0], mesh_inds[1], mesh_inds[2]] - np.floor(shape[0] / 2)),
            int(self.data_meshes[source_idx][sample_type]['y'][mesh_inds[0], mesh_inds[1], mesh_inds[2]] - np.floor(shape[1] / 2)),
            int(self.data_meshes[source_idx][sample_type]['z'][mesh_inds[0], mesh_inds[1], mesh_inds[2]] - np.floor(shape[2] / 2)),
        ]
        bbox = origin + list(shape)

        return source_idx, bbox

    def get_source_mesh_for_sample_idx(self, sample_idx):
        # Get appropriate training data cube sample_idx based on global linear sample_idx
        source_idx = int(np.argmax(np.asarray(self.data_inds_max) >= int(sample_idx)))
        # Get appropriate subscript index for the respective training data cube, given the global linear index
        mesh_inds = np.unravel_index(sample_idx - self.data_inds_min[source_idx],
                                     dims=self.data_meshes[source_idx]['target']['x'].shape)

        return source_idx, mesh_inds

    def get_datasources_stats(self, num_samples=30):
        return [self.get_datasource_stats(i, num_samples) for i in range(len(self.data_sources))]

    def get_datasource_stats(self, data_source_idx, num_samples=30):
        sample_inds = np.random.random_integers(self.data_inds_min[data_source_idx],
                                                self.data_inds_max[data_source_idx], num_samples)
        means = []
        stds = []
        for i, sample_idx in enumerate(sample_inds):
            print('Getting stats from dataset ... sample {} of {}'.format(i, num_samples))
            data = self.get_ordered_sample(sample_idx, normalize=False)
            means.append(np.mean(data['input'].data.numpy()))
            stds.append(np.std(data['input'].data.numpy()))

        return {'mean': float(np.around(np.mean(means), 1)), 'std': float(np.around(np.mean(stds), 1))}

    def update_datasources_stats(self, num_samples=30):
        [self.update_datasource_stats(i, num_samples) for i in range(len(self.data_sources))]

    def update_datasource_stats(self, data_source_idx, num_samples=30):

        stats = self.get_datasource_stats(data_source_idx, num_samples)
        self.data_sources[data_source_idx] = self.data_sources[data_source_idx]._replace(input_mean=stats['mean'])
        self.data_sources[data_source_idx] = self.data_sources[data_source_idx]._replace(input_std=stats['std'])

    def datasource_id_to_idx(self, id):
        idx = [data_source.id for data_source in self.data_sources].index(id)
        return idx

    def datasource_idx_to_id(self, idx):
        id = self.data_sources[idx].id
        return id

    def show_sample(self, sample_idx):
        (data, index) = self.__getitem__(sample_idx)
        fig, axs = plt.subplots(1,2)
        input_ = data['input'].data.numpy().squeeze()
        axs[0].imshow(input_, cmap='gray')
        target = data['target'].data.numpy().squeeze()
        while target.ndim < 2:
            target = np.expand_dims(target, 0)
        axs[1].imshow(target, cmap='gray')

    @staticmethod
    def collate_fn(batch):
        input_ = torch.cat([torch.unsqueeze(item['input'], dim=0) for item in batch], dim=0)
        target = torch.cat([torch.unsqueeze(item['target'], dim=0) for item in batch], dim=0)
        sample_idx = [item['sample_idx'] for item in batch]
        return {'input': input_, 'target': target, 'sample_idx': sample_idx}

    @staticmethod
    def normalize(data, mean, std):
        return (np.asarray(data) - mean) / std

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
    def wkw_create(wkw_path, wkw_dtype=np.uint8):
        wkw.Dataset.create(wkw_path, wkw.Header(np.uint8))

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

        return np.floor(total_size/1024/1024)  # MiB

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
