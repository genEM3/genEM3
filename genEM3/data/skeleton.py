"""Functions for manipulating the skeleton object (These could be moved to wkskel repo if not already there)"""
import os
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
import wkskel
from wkskel import Skeleton

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir


def getAllTreeCoordinates(skel):
    """Get the coordinates of all the nodes in trees within a skeleton object"""
    # Get the coordinate of the nodes as numpy array
    coordsList = [skel.nodes[x].position.values for x in range(skel.num_trees())]
    # Concatenate into a single array of coordinates
    return np.vstack(coordsList)


def get_volume_df(skeletons: Sequence[Skeleton]):
    """Function to return a table containing the information from test box annotations
    Arguments:
        skeletons: a list of skeletons of the annotated test boxes
    Returns:
        volume_df: pandas data frame which contains the skeleton id, tree_idx and id, coordinate 
            of the center of the patches and their relative location (xi, yi). Finally, it contains
            the class of the annotated patch(debris(0, False) vs. clean(1, True))
    """    
    volume_df = pd.DataFrame(columns=['skel_idx', 'tree_idx', 'tree_id',
                                      'x', 'y', 'z', 'xi', 'yi', 'class', 'explicit'])
    for skel_idx, skel in enumerate(skeletons):
        group_ids = np.array(skel.group_ids)
        for plane_group in skel.groups:
            plane_group_id = plane_group.id
            plane_group_class = bool(int(plane_group.name[-1]))
            plane_tree_inds = np.where(group_ids == plane_group_id)[0]
            plane_matrix = np.zeros((5, 5), dtype=np.bool)
            plane_df = pd.DataFrame(columns=['skel_idx', 'tree_idx', 'tree_id',
                                             'x', 'y', 'z', 'xi', 'yi', 'class', 'explicit'])
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
                plane_df = plane_df.append({
                    'skel_idx': skel_idx,
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
    return volume_df


def add_bbox_tree_from_center(coord_center, input_shape, tree_name, skel):
    """Adds a bbox skeleton at specified coordinate to skeleton""" 
    cx, cy, cz = coord_center
    positions = np.array([
        [cx, cy, cz],
        [cx - input_shape[0]/2, cy - input_shape[1]/2, cz],
        [cx - input_shape[0]/2, cy + input_shape[1]/2, cz],
        [cx + input_shape[0]/2, cy - input_shape[1]/2, cz],
        [cx + input_shape[0]/2, cy + input_shape[1]/2, cz],
        [cx + input_shape[0]/2, cy + input_shape[1]/2 - 1, cz]
    ])
    min_id = skel.max_node_id() + 1
    max_id = min_id + positions.shape[0] - 1
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
    skel.add_tree(
        nodes=nodes,
        edges=edges,
        name=tree_name)


def make_skel_from_json(json_path: str):
    """
    Creates a skeleton object from the binary targets of the data sources in json
    Args:
        json_path: the path of the data source json file
    Returns:
        skel: the skeleton object
    """
    data_sources_dict = WkwData.convert_ds_to_dict(WkwData.read_short_ds_json(json_path=json_path))
    # Init with empty skeleton
    empty_skel_name = os.path.join(get_data_dir(), 'NML', 'empty_skel.nml')
    skel = wkskel.Skeleton(nml_path=empty_skel_name)

    # Loop over each bbox
    keys = list(data_sources_dict.keys())
    num_nodes_perTree = 5
    for idx, key in tqdm(enumerate(keys), desc='Making bbox nml', total=len(keys)):
        # Get minimum and maximum node id
        min_id = (num_nodes_perTree * idx) + 1
        max_id = num_nodes_perTree * (idx + 1)
        # Encode the target in the tree name
        cur_target = data_sources_dict[key]['target_class']
        cur_name = f'{key}, Debris: {cur_target[0]}, Myelin: {cur_target[1]}'
        # add current tree
        add_bbox_tree(skel=skel,
                      bbox=data_sources_dict[key]['input_bbox'],
                      tree_name=cur_name,
                      node_id_min_max=[min_id, max_id])
    return skel


def add_bbox_tree(skel, bbox: list, tree_name: str, node_id_min_max: list):
    """
    Get the nodes and edges of a bounding box tree
    """
    corners = corners_from_bbox(bbox=bbox)
    min_id, max_id = node_id_min_max
    # Nodes
    nodes = skel.define_nodes(
        position_x=corners[:, 0].tolist(),
        position_y=corners[:, 1].tolist(),
        position_z=corners[:, 2].tolist(),
        id=list(range(min_id, max_id + 1))
    )
    # Edges
    edges = np.array([
        [1, 2],
        [2, 4],
        [4, 3],
        [3, 5]
    ]) + min_id - 1
    # Add tree
    # Note: There's no need to return the object since the change is not limited to the scope of the function
    skel.add_tree(nodes=nodes, edges=edges, name=tree_name)


def corners_from_bbox(bbox: list):
    """
    Get the coordinates of the corners given a webknossos style bounding box
    Args:
        bbox: 1 x 6 vector of webknossos bbox: upper left corner + bbox shape
    Returns:
        corners: 5 x 3 coordinates of the corners of the 2D image patch. 
        Upper left corner is repeated to close the loop
    """ 
    # Get upper left corner and shape of bbox
    upper_left = np.asarray(bbox[0:3])
    shape = np.asarray(bbox[3:])
    # no change in Z
    shape[2] = 0
    # Get the shift for the corners of the bbox
    zeros = np.zeros((1, 3), dtype=np.int64)
    deltas = np.vstack((zeros, shape, shape, shape, zeros))
    deltas[[1, 2], [1, 0]] = 0
    # Setup corners
    corners = np.tile(upper_left, (5, 1))
    corners = corners + deltas
    return corners
