import os

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir

# Read Json file
json_names = ['dense_3X_10_10_2_um/original_merged_double_binary_v01.json', 
             '10x_test_bboxes/10X_9_9_1_um_double_binary_v01.json']
ds_names = [os.path.join(get_data_dir(), j_name) for j_name in json_names]
data_sources = []
dataset_path = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8_artifact_pred/color/1'
for ds in ds_names:
    cur_ds = WkwData.datasources_from_json(json_path=ds)
    cur_ds_dict = WkwData.convert_ds_to_dict(cur_ds)
    # all pathes use the artifact_pred dataset
    for s in cur_ds_dict:
        cur_source = cur_ds_dict[s]
        cur_source['input_path'] = dataset_path
        cur_source['target_path'] = dataset_path
        cur_ds_dict[s] = cur_source
    # Write out the jsons
    cur_ds_corrected_list = WkwData.convert_ds_to_list(datasources_dict=cur_ds_dict)
    WkwData.datasources_to_json(datasources=cur_ds_corrected_list, json_path=ds)
