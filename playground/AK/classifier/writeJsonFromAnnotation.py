import os
# Make sure that the X-forwarding doesn't throw an error
import matplotlib
matplotlib.use('Agg')

from genEM3.util.path import get_data_dir, getMag8DatasetDir
from genEM3.data.annotation import Widget, remove_bbox_margin
from genEM3.data.wkwdata import WkwData, DataSource

# Load the annotation widget
file_name = '/u/alik/code/genEM3/playground/AK/classifier/.log/10X_9_9_1_um_with_myelin_Final.pkl'
w_loaded = Widget.load(file_name=file_name)
# Get the datasources
source_list = []
sources_fromWidget = w_loaded.dataset.data_sources
for i, cur_source in enumerate(sources_fromWidget):
    # correct the bbox back to the original bbox
    # fix shape
    cur_input_bbox = remove_bbox_margin(cur_source.input_bbox, margin=35)
    cur_target_bbox = remove_bbox_margin(cur_source.target_bbox, margin=35)
    # Update the binary targets to two binary decisions for the presence of image artefacts and Myelin
    cur_targets = [w_loaded.annotation_list[i][1].get('Debris'), w_loaded.annotation_list[i][1].get('Myelin')]
    source_list.append(DataSource(id=cur_source.id, input_path=getMag8DatasetDir(), input_bbox=cur_input_bbox,
                                  input_mean=cur_source.input_mean, input_std=cur_source.input_std, target_path=getMag8DatasetDir(),
                                  target_bbox=cur_target_bbox, target_class=cur_targets, target_binary=cur_source.target_binary))
    
# Json name
json_name = os.path.join(get_data_dir(), '10x_test_bboxes', '10X_9_9_1_um_double_binary_v01.json')
# Write to json file
WkwData.datasources_to_json(source_list, json_name)