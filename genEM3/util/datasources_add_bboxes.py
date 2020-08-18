import os
import numpy as np
from genEM3.data.wkwdata import WkwData, DataSource

run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, '../../data/debris_clean_datasource.json')
datasources_json_path_out = os.path.join(run_root, '../../data/debris_clean_added_bboxes_datasource.json')

bboxes_add = [
    [25000, 20000, 3510, 560, 560, 5],
    [25000, 20000, 3533, 560, 560, 12],
    [25000, 20000, 3548, 560, 560, 6],
    [25000, 20000, 3636, 560, 560, 15],
    [25000, 20000, 3680, 560, 560, 13],
    [25000, 20000, 3696, 560, 560, 11],
    [25000, 20000, 3776, 560, 560, 4],
    [25000, 20000, 3810, 560, 560, 40]
]
target_binary_add = 1
target_class_add = 0.0
input_mean_add = 148.0
input_std_add = 36.0
path_add = "/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1"

data_sources = WkwData.datasources_from_json(datasources_json_path)
data_sources_max_id = max([int(data_source.id) for data_source in data_sources])

data_sources_out = data_sources
for bbox_idx, bbox_add in enumerate(bboxes_add):
    data_source_out = DataSource(
        id=str(data_sources_max_id + bbox_idx + 1),
        input_path=path_add,
        input_bbox=bbox_add,
        input_mean=input_mean_add,
        input_std=input_std_add,
        target_path=path_add,
        target_bbox=bbox_add,
        target_class=target_class_add,
        target_binary=target_binary_add
    )
    data_sources_out.append(data_source_out)

WkwData.datasources_to_json(data_sources_out, datasources_json_path_out)

