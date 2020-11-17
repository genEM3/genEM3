"""
Functions used in relation to data annotation that might not fit in other modules
"""
import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from genEM3.data.wkwdata import DataSource, WkwData
from genEM3.util.path import get_data_dir


def update_data_source_targets(dataset: WkwData,
                               target_index_tuple_list: Sequence[Tuple[int, float]]):
    """Create an updated list of datasources from a wkwdataset and a list of sample index, target_class pair"""
    list_source_idx = [dataset.get_source_idx_from_sample_idx(sample_idx) for (sample_idx, _) in target_index_tuple_list]
    source_list = []
    for i, cur_target_tuple in enumerate(target_index_tuple_list):
        cur_target = cur_target_tuple[1]
        sample_idx = cur_target_tuple[0]
        s_index = list_source_idx[sample_idx]
        s = dataset.data_sources[s_index]
        source_list.append(DataSource(id=s.id, input_path=s.input_path, input_bbox=s.input_bbox,
                                      input_mean=s.input_mean, input_std=s.input_std, target_path=s.target_path,
                                      target_bbox=s.target_bbox, target_class=cur_target, target_binary=s.target_binary))
    return source_list


def update_data_source_bbox(dataset: WkwData,
                            bbox_list: Sequence[Tuple[int, Sequence[int]]]):
    """Create an updated list of datasources from a wkwdataset and a list of index and bounding box tuples"""
    assert len(bbox_list) == len(dataset)
    source_list = []
    for sample_idx, (source_idx, cur_bbox) in enumerate(bbox_list):
        s = dataset.data_sources[source_idx]
        source_list.append(DataSource(id=str(sample_idx), input_path=s.input_path, input_bbox=cur_bbox,
                                      input_mean=s.input_mean, input_std=s.input_std, target_path=s.target_path,
                                      target_bbox=cur_bbox, target_class=s.target_class, target_binary=s.target_binary))
    return source_list


def display_example(index: int, dataset: WkwData, margin: int = 35, roi_size: int = 140):
    """Display an image with a central rectangle for the roi"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dataset.get_ordered_sample(index)['input'].squeeze(),cmap='gray')
    rectangle = plt.Rectangle((margin,margin), roi_size, roi_size, fill=False, ec="red")
    ax.add_patch(rectangle)
    plt.show()


def merge_json_from_data_dir(fnames: Sequence[str], output_fname: str):
    """Function concatenates the data directory to the list of file names and concatenats the related jsons"""
    # Test concatenating jsons
    full_fnames = []
    for fname in fnames:
        full_fname = os.path.join(get_data_dir(), fname)
        full_fnames.append(full_fname)

    # Concatenate the test and training data sets
    full_output_name = os.path.join(get_data_dir(), output_fname)
    all_ds = WkwData.concat_datasources(json_paths_in=full_fnames, json_path_out=full_output_name)
    return all_ds


def patch_source_list_from_dataset(dataset: WkwData, 
                                   margin: int = 35,
                                   roi_size: int = 140):
    """Return two data_sources from the image patches contained in a dataset. One data source has a larger bbox for annotations"""
    corner_xy_index = [0, 1]
    length_xy_index = [3, 4]
    large_bboxes_idx = []
    bboxes_idx = []
    for idx in range(len(dataset)):
        (source_idx, original_cur_bbox) = dataset.get_bbox_for_sample_idx(idx)
        bboxes_idx.append((source_idx, original_cur_bbox))
        cur_bbox = np.asarray(original_cur_bbox)
        cur_bbox[corner_xy_index] = cur_bbox[corner_xy_index] - margin
        cur_bbox[length_xy_index] = cur_bbox[length_xy_index] + margin*2
        # large bbox append
        large_bboxes_idx.append((source_idx, cur_bbox.tolist()))

    assert len(large_bboxes_idx) == len(dataset) == len(bboxes_idx)
    larger_sources = update_data_source_bbox(dataset, large_bboxes_idx)
    patch_source_list = update_data_source_bbox(dataset, bboxes_idx)
    return {'original': patch_source_list,'large':larger_sources}


def divide_range(total_size: int, chunk_size: int = 1000,):
    """Break down the range into partitions of 1000"""
    chunk_size = 1000
    num_thousand, remainder = divmod(total_size, chunk_size)
    list_ranges = []
    # Create a list of ranges
    for i in range(num_thousand):
        list_ranges.append(range(i*chunk_size, (i+1)*chunk_size))
    if remainder > 0:
        final_range = range(num_thousand*chunk_size, num_thousand*chunk_size+remainder)
        list_ranges.append(final_range)

    return list_ranges
