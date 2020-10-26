import os
import pandas as pd
import numpy as np
from scipy.ndimage.measurements import label
from wkskel import Skeleton, Parameters, Nodes
from genEM3.data.wkwdata import WkwData, DataSource
from genEM3.training.metrics import Metrics
from genEM3.util.path import get_runs_dir

path_in = os.path.join(get_runs_dir(), 'inference/ae_classify_11_parallel/test_center_filt')
cache_HDD_root = os.path.join(path_in, '.cache/')
path_datasources = os.path.join(path_in, 'datasources.json')
path_nml_in = os.path.join(path_in, 'bbox_annotated.nml')
input_shape = (140, 140, 1)
target_shape = (1, 1, 1)
stride = (35, 35, 1)

datasources = WkwData.datasources_from_json(path_datasources)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=target_shape,
    data_sources=datasources,
    stride=stride,
    cache_HDD=False,
    cache_RAM=True)

skel = Skeleton(path_nml_in)

pred_df = pd.DataFrame(columns=['tree_idx', 'tree_id', 'x', 'y', 'z', 'xi', 'yi', 'class', 'explicit', 'cluster_id', 'prob'])
group_ids = np.array(skel.group_ids)
input_path = datasources[0].input_path
input_bbox = datasources[0].input_bbox
structure = np.ones((3, 3), dtype=np.int)
cluster_id = 0
for plane_group in skel.groups:
    plane_group_id = plane_group.id
    plane_group_class = bool(int(plane_group.name[-1]))
    plane_tree_inds = np.where(group_ids == plane_group_id)[0]
    plane_matrix = np.zeros((5, 5), dtype=np.bool)
    plane_df = pd.DataFrame(columns=['tree_idx', 'tree_id', 'x', 'y', 'z', 'xi', 'yi', 'class', 'explicit', 'cluster_id', 'prob'])
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
        prob = dataset.data_cache_input[input_path][str(input_bbox)][0, c_rel[0], c_rel[1], c_rel[2]]
        plane_df = plane_df.append({
            'tree_idx': tree_idx,
            'tree_id': skel.tree_ids[tree_idx],
            'x': c_abs[0],
            'y': c_abs[1],
            'z': c_abs[2],
            'xi': patch_xi,
            'yi': patch_yi,
            'class': patch_class,
            'explicit': explicit,
            'cluster_id': None,
            'prob': prob
        }, ignore_index=True)

    comp_labeled, n_comp = label(plane_matrix, structure)
    if n_comp > 0:
        for comp_idx in range(1, n_comp + 1):
            cluster_id = cluster_id + 1
            xis, yis = np.where(comp_labeled == comp_idx)
            for xi, yi in zip(xis, yis):
                plane_df.loc[(plane_df['xi'] == xi) & (plane_df['yi'] == yi), 'cluster_id'] = cluster_id

    pred_df = pred_df.append(plane_df)

# Write table
pred_df.to_csv(os.path.join(path_in, 'patch_test_data.csv'))

# Metrics for single patches
metrics = Metrics(
    targets=pred_df['class'],
    outputs=pred_df['prob'],
    output_prob_fn=lambda x: x)

metrics.pr_curve(n_steps=100, path_out=os.path.join(path_in, 'patch_pr.csv'))
metrics.confusion_table(path_out=os.path.join(path_in, 'patch_confusion.csv'))

# Metrics for clusters
pred_df_cluster = pred_df.copy(deep=True)
for cluster_id in pred_df['cluster_id'].unique():
    if (cluster_id is not None):
        if (cluster_id > 0):
            x_avg = np.mean(pred_df_cluster.loc[pred_df_cluster['cluster_id'] == cluster_id, 'x'])
            y_avg = np.mean(pred_df_cluster.loc[pred_df_cluster['cluster_id'] == cluster_id, 'y'])
            z_avg = np.mean(pred_df_cluster.loc[pred_df_cluster['cluster_id'] == cluster_id, 'z'])
            prob_max = np.max(pred_df_cluster.loc[pred_df_cluster['cluster_id'] == cluster_id, 'prob'])
            pred_df_cluster = pred_df_cluster[~(pred_df_cluster['cluster_id'] == cluster_id)]
            pred_df_cluster = pred_df_cluster.append({
                'tree_idx': None,
                'tree_id': None,
                'x': x_avg,
                'y': y_avg,
                'z': z_avg,
                'xi': None,
                'yi': None,
                'class': True,
                'explicit': None,
                'cluster_id': cluster_id,
                'prob': prob_max}, ignore_index=True)

pred_df_cluster.to_csv(os.path.join(path_in, 'cluster_test_data.csv'))

metrics = Metrics(
    targets=pred_df_cluster['class'],
    outputs=pred_df_cluster['prob'],
    output_prob_fn=lambda x: x)

metrics.pr_curve(n_steps=100, path_out=os.path.join(path_in, 'cluster_pr.csv'))
metrics.confusion_table(path_out=os.path.join(path_in, 'cluster_confusion.csv'))
