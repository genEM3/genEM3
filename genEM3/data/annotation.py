"""
Functions used in relation to data annotation that might not fit in other modules
"""
import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt

from genEM3.data.wkwdata import DataSource, WkwData
from genEM3.util.path import getDataDir

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
    assert len(bbox_list)==len(dataset)
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
        full_fname = os.path.join(getDataDir(), fname)
        full_fnames.append(full_fname)

    # Concatenate the test and training data sets
    full_output_name = os.path.join(getDataDir(), output_fname)
    all_ds = WkwData.concat_datasources(json_paths_in=full_fnames, json_path_out=full_output_name)
    return all_ds
