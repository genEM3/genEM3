import os

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir

# Read Json file
f_name = 'dense_3X_10_10_2_um/original_merged_double_binary_v01.json'
json_path = os.path.join(get_data_dir(), f_name)
data_sources = WkwData.datasources_from_short_json(json_path=json_path)
# Write test json
small_ds = data_sources[0:2]
small_ds = WkwData.convert_to_short_ds(small_ds)
WkwData.datasources_to_short_json(datasources=small_ds, json_path='test.json')
reloaded_ds = WkwData.datasources_from_short_json(json_path='test.json')