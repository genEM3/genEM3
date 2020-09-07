import os
import numpy as np
from wkskel import Skeleton, Parameters, Nodes
from genEM3.data.wkwdata import WkwData, DataSource

bbox_val_dims_um = (10, 10, 2)
scale = (32, 32, 35)
input_shape = (140, 140, 1)
target_shape = (1, 1, 1)
stride = (35, 35, 1)
path_in = '/home/drawitschf/Code/genEM3/runs/inference/ae_classify_11_parallel/test_top'
path_datasources = os.path.join(path_in, 'datasources.json')
path_nml_out = os.path.join(path_in, 'bbox.nml')

datasources = WkwData.datasources_from_json(path_datasources)
dataset = WkwData(
    input_shape=input_shape,
    target_shape=target_shape,
    data_sources=datasources,
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

skel = Skeleton('/home/drawitschf/Code/genEM3/runs/inference/ae_classify_11_parallel/empty.nml')
min_id = 1
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
    max_id = min_id + positions.shape[0] - 1
    # nodes = skel.define_nodes_from_positions(positions)
    nodes = skel.define_nodes(
        position_x=positions[:, 0].tolist(),
        position_y=positions[:, 1].tolist(),
        position_z=positions[:, 2].tolist(),
        id=list(range(min_id, max_id + 1))
    )
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
        skel.add_group(id=zi, name='plane{:03d}:X'.format(zi))

    skel.add_tree(
        nodes=nodes,
        edges=edges,
        name='patch{:02d}.{:02d}:X'.format(xi, yi),
        group_id=zi)

    min_id = max_id + 1


skel.write_nml(path_nml_out)





