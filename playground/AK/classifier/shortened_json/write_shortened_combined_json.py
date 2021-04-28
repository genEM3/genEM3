import os

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir

# Read Json file
json_names = ['dense_3X_10_10_2_um/original_merged_double_binary_v01.json', 
             '10x_test_bboxes/10X_9_9_1_um_double_binary_v01.json']
ds_names = [os.path.join(get_data_dir(), j_name) for j_name in json_names]
data_sources = WkwData.concat_datasources(ds_names)
# Get the short version of the data sources
output_name = os.path.join(get_data_dir(), 'combined', 'combined_20K_patches.json')
short_ds = WkwData.convert_to_short_ds(data_sources=data_sources)
# Write combined data source json file
WkwData.write_short_ds_json(datasources=short_ds, json_path=output_name)