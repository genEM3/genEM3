import os
# Make sure that the X-forwarding doesn't throw an error
import matplotlib
matplotlib.use('Agg')

from genEM3.util.path import get_data_dir
from genEM3.data.annotation import Widget
from genEM3.data.wkwdata import WkwData, DataSource

# Load the annotation widget
file_name = '/u/alik/code/genEM3/playground/AK/classifier/.log/original_merged_with_myelin_Final.pkl'
w_loaded = Widget.load(file_name=file_name)
# Get the datasources
source_list = []
sources_fromWidget = w_loaded.dataset.data_sources
for i, cur_source in enumerate(sources_fromWidget):
    # correct the bbox back to 
    cur_source.input_bbox[3:5] = [140, 140]
    cur_source.target_bbox[3:5] = [140, 140]
    # Update the binary targets to two binary decisions for the presence of image artefacts and Myelin
    cur_targets = [w_loaded.annotation_list[i][1].get('Debris'), w_loaded.annotation_list[i][1].get('Myelin')]
    source_list.append(DataSource(id=cur_source.id, input_path=cur_source.input_path, input_bbox=cur_source.input_bbox,
                              input_mean=cur_source.input_mean, input_std=cur_source.input_std, target_path=cur_source.target_path,
                              target_bbox=cur_source.target_bbox, target_class=cur_targets, target_binary=cur_source.target_binary))
    
# Json name
json_name = os.path.join(get_data_dir(), 'dense_3X_10_10_2_um', 'original_merged_double_binary_v01.json')
# Write to json file
WkwData.datasources_to_json(source_list, json_name)