"""
Functions used in relation to data annotation that might not fit in other modules
"""
from genEM3.data.wkwdata import DataSource, WkwData
from typing import Sequence, Tuple


def update_data_source_targets(dataset: WkwData, target_index_tuple_list: Sequence[Tuple[int, float]]):
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
