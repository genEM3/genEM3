import os
import numpy as np
from genEM3.data.wkwdata import WkwData, DataSource

run_root = os.path.dirname(os.path.abspath(__file__))
datasources_json_path = os.path.join(run_root, '../../data/debris_clean_datasource.json')
datasources_json_path_out = os.path.join(run_root, '../../data/debris_clean_wiggle_datasource.json')


wiggles = [
    [-35, 0],
    [35, 0],
    [0, -35],
    [0, 35]
]

data_sources = WkwData.datasources_from_json(datasources_json_path)

data_sources_out = []
for data_source in data_sources:
    data_sources_out.append(data_source)
    id = data_source.id
    if data_source.target_binary == 1:
        bbox = data_source.input_bbox
        for wiggle_idx, wiggle in enumerate(wiggles):
            id_out = '{:05.0f}'.format(int(id)+(wiggle_idx+1)*1E4)
            bbox_out = [bbox[0] + wiggle[0], bbox[1] + wiggle[1], *bbox[2:]]
            data_source_out = DataSource(
                id=id_out,
                input_path=data_source.input_path,
                input_bbox=bbox_out,
                input_mean=data_source.input_mean,
                input_std=data_source.input_std,
                target_path=data_source.target_path,
                target_bbox=bbox_out,
                target_class=data_source.target_class,
                target_binary=data_source.target_binary
            )
            data_sources_out.append(data_source_out)

WkwData.datasources_to_json(data_sources_out, datasources_json_path_out)

