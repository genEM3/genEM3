import os
import time

import numpy as np

from wkskel import Skeleton
from genEM3.data.wkwdata import WkwData, DataSource
from genEM3.util.path import get_runs_dir, getDataDir
from genEM3.util.image import bboxesFromArray
from genEM3.data.skeleton import get_volume_df

# Get the names of three test skeletons
path_in_stub = os.path.join(get_runs_dir(), 'inference/ae_classify_11_parallel')
test_dirs = ['test_center_filt', 'test_bottom_filt', 'test_top_filt']
skel_dirs = [os.path.join(path_in_stub, d, 'bbox_annotated.nml') for d in test_dirs]
# check that all of the files exist
assert all([os.path.exists(skel_dir) for skel_dir in skel_dirs])

# Create skeleton objects
start = time.time()
skeletons = [Skeleton(skel_dir) for skel_dir in skel_dirs]
print(f'Time to read skeleton: {time.time() - start}')
# Read the coordinates and target class of all three skeletons into the volume data frame
volume_df = get_volume_df(skeletons=skeletons)
# Get the ingredients for making the datasources
bboxes = bboxesFromArray(volume_df[['x', 'y', 'z']].values)
input_dir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8_artifact_pred/color/1'
target_class = volume_df['class'].values.astype(np.float)
target_binary = 1
target_dir = input_dir
input_mean = 148.0
input_std = 36.0
# Create a list of data sources
source_list = []
for i, cur_bbox in enumerate(bboxes):
    cur_target = target_class[i]
    source_list.append(DataSource(id=str(i), input_path=input_dir, input_bbox=cur_bbox.tolist(),
                                  input_mean=input_mean, input_std=input_std, target_path=target_dir,
                                  target_bbox=cur_bbox.tolist(), target_class=cur_target, target_binary=target_binary))
# Json name
json_name = os.path.join(getDataDir(), 'test_data_three_bboxes.json')
# Write to json file
WkwData.datasources_to_json(source_list, json_name)
