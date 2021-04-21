import os
from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir
# Read the data
json_name = os.path.join(get_data_dir(), 'combined', 'combined_20K_patches.json')
data_sources = WkwData.read_short_ds_json(json_path=json_name)
# Read an old json for comparison
old_json_name = os.path.join(get_data_dir(), 'dense_3X_10_10_2_um/original_merged_double_binary_v01.json')
old_example = WkwData.datasources_from_json(old_json_name)
breakpoint()