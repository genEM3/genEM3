import os
import numpy as np
from wkskel import Skeleton, Parameters, Nodes
from genEM3.data.wkwdata import WkwData, DataSource


def bbox_from_wkw(wkw_z, wkw_y, wkw_x):
    return [wkw_x * 1024, wkw_y * 1024, wkw_z * 1024, 1024, 1024, 1024]

bbox_val_dims_um = (10, 10, 1)
bbox = np.array(bbox_from_wkw(1, 15, 20))
bbox_str = '_'.join([str(b) for b in bbox])
nml_path_out = '/home/drawitschf/Code/genEM3/runs/inference/ae_classify_11_parallel/test_top/bbox_' + bbox_str + '.nml'

path = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8_artifact_pred/color/1'
name = os.path.split(os.path.split(os.path.split(path)[0])[0])[1]

datasource = DataSource(id=0, input_path=path, input_bbox=bbox, target_path=path, target_bbox=bbox)

scale = (32, 32, 35)
input_shape = (140, 140, 1)
target_shape = (1, 1, 1)
stride = (35, 35, 1)

dataset = WkwData(
    input_shape=input_shape,
    target_shape=target_shape,
    data_sources=[datasource],
    stride=stride,
    cache_HDD=False,
    cache_RAM=False)

bbox_val_dims_vx = np.array(bbox_val_dims_um) * 1000 / np.array(scale)
n_fits = np.ceil((bbox_val_dims_vx - np.array(input_shape)) / np.array(stride)).astype(int)
meshes = dataset.data_meshes[0]['input']
meshes_shape = np.array(meshes['x'].shape)
meshes_center = np.floor(meshes_shape / 2)
meshes_min = np.floor(meshes_center - n_fits/2).astype(int)
meshes_max = np.floor(meshes_center + n_fits/2).astype(int)

meshes_val = {key: meshes[key][meshes_min[0]:meshes_max[0], meshes_min[1]:meshes_max[1], meshes_min[2]:meshes_max[2]] for key in meshes}

parameters = Parameters(name=name, scale=scale)
# skel = Skeleton(parameters=parameters)
skel = Skeleton('/home/drawitschf/Code/genEM3/runs/inference/ae_classify_11_parallel/empty.nml')

for idx in range(np.prod(n_fits)):
    print('adding trees {}/{}'.format(idx, np.prod(n_fits)))
    xi, yi, zi = np.unravel_index(idx, shape=n_fits)
    cx = meshes_val['x'][xi, yi, zi]
    cy = meshes_val['y'][xi, yi, zi]
    cz = meshes_val['z'][xi, yi, zi]
    positions = np.array([
        [cx, cy, cz],
        [cx - input_shape[0]/2, cy - input_shape[1]/2, cz],
        [cx - input_shape[0]/2, cy + input_shape[1]/2, cz],
        [cx + input_shape[0]/2, cy - input_shape[1]/2, cz],
        [cx + input_shape[0]/2, cy + input_shape[1]/2, cz],
        [cx + input_shape[0]/2, cy + input_shape[1]/2 - 1, cz]
    ])
    nodes = skel.define_nodes_from_positions(positions)
    edges = np.array([
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 4],
        [3, 5],
        [4, 6]
    ]) + skel.max_node_id()
    if zi not in skel.group_ids:
        skel.add_group(id=zi, name='plane{}:X'.format(zi))

    skel.add_tree(nodes=nodes, edges=edges, group_id=zi, name='patch{}:X'.format(idx))

skel.write_nml(nml_path_out)



