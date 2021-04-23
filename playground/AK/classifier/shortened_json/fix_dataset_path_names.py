import os

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir

# Read Json file
json_names = ['dense_3X_10_10_2_um/original_merged_double_binary_v01.json', 
             '10x_test_bboxes/10X_9_9_1_um_double_binary_v01.json']
ds_names = [os.path.join(get_data_dir(), j_name) for j_name in json_names]