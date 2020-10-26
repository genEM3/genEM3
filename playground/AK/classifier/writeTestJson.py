import os
import time

import pandas as pd
import numpy as np

from wkskel import Skeleton, Parameters, Nodes
from genEM3.data.wkwdata import WkwData, DataSource
from genEM3.util.path import get_runs_dir, getDataDir
from genEM3.util.image import bboxesFromArray
path_in = os.path.join(get_runs_dir(), 'inference/ae_classify_11_parallel/test_center_filt')
cache_HDD_root = os.path.join(path_in, '.cache/')
path_nml_in = os.path.join(path_in, 'bbox_annotated.nml')
# Get the input_bbox used to get relative coordinate
path_datasources = os.path.join(path_in, 'datasources.json')
datasources = WkwData.datasources_from_json(path_datasources)
input_bbox = datasources[0].input_bbox

start = time.time()
skel = Skeleton(path_nml_in)
print(f'Time to read skeleton: {time.time() - start}')

volume_df = pd.DataFrame(columns=['tree_idx', 'tree_id', 'x', 'y', 'z', 'xi', 'yi', 'class', 'explicit'])
group_ids = np.array(skel.group_ids)

for plane_group in skel.groups:
    plane_group_id = plane_group.id
    plane_group_class = bool(int(plane_group.name[-1]))
    plane_tree_inds = np.where(group_ids == plane_group_id)[0]
    plane_matrix = np.zeros((5, 5), dtype=np.bool)
    plane_df = pd.DataFrame(columns=['tree_idx', 'tree_id', 'x', 'y', 'z', 'xi', 'yi', 'class', 'explicit'])

    for tree_idx in plane_tree_inds:
        patch_class = skel.names[tree_idx][-1]
        if patch_class.isnumeric():
            patch_class = bool(int(patch_class))
            explicit = True
        else:
            patch_class = plane_group_class
            explicit = False

        patch_xi = int(skel.names[tree_idx][5:7])
        patch_yi = int(skel.names[tree_idx][8:10])
        plane_matrix[patch_xi, patch_yi] = patch_class

        c_id = np.argmax(np.bincount(skel.edges[tree_idx].flatten()))
        c_abs = skel.nodes[tree_idx].loc[skel.nodes[tree_idx]['id'] == c_id, 'position'].values[0].astype(int)
        c_rel = c_abs - np.array(input_bbox[0:3])
        plane_df = plane_df.append({
            'tree_idx': tree_idx,
            'tree_id': skel.tree_ids[tree_idx],
            'x': c_abs[0],
            'y': c_abs[1],
            'z': c_abs[2],
            'xi': patch_xi,
            'yi': patch_yi,
            'class': patch_class,
            'explicit': explicit
        }, ignore_index=True)
    
    volume_df = volume_df.append(plane_df)
# Get the bounding boxes
bboxes = bboxesFromArray(volume_df[['x', 'y', 'z']].values)
input_dir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8_artifact_pred/color/1'
target_class = volume_df['class'].values.astype(np.float)
target_binary = 1
target_dir = input_dir
input_mean = 148.0
input_std = 36.0

source_list = []
for i, cur_bbox in enumerate(bboxes):
    cur_target = target_class[i]
    source_list.append(DataSource(id=str(i), input_path=input_dir, input_bbox=cur_bbox.tolist(),
                                  input_mean=input_mean, input_std=input_std, target_path=target_dir,
                                  target_bbox=cur_bbox.tolist(), target_class=cur_target, target_binary=target_binary))
json_name = os.path.join(getDataDir(), 'test_box_center.json')
WkwData.datasources_to_json(source_list, json_name)
