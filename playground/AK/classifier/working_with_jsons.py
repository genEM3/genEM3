import os

from genEM3.data.wkwdata import Wkwdata
json_names = 'dense_3X_10_10_2_um/original_merged_double_binary_v01.json'
ds_names = os.path.join(get_data_dir(), j_name) 
data_sources = WkwData.data  (ds_names)