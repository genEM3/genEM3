"""Functions for manipulating the skeleton object (These could be moved to wkskel repo if not already there)"""
from typing import Sequence
import numpy as np
import pandas as pd

from wkskel import Skeleton


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


def add_bbox_tree(skel, bbox: list, tree_name: str):
    """
    Add a tree based on the bounding box
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
    # Create nodes and edges from corners
    min_id = skel.max_node_id() + 1
    max_id = min_id + corners.shape[0] - 1
    nodes = skel.define_nodes(
        position_x=corners[:, 0].tolist(),
        position_y=corners[:, 1].tolist(),
        position_z=corners[:, 2].tolist(),
        id=list(range(min_id, max_id + 1))
    )
    edges = np.array([
        [1, 2],
        [2, 4],
        [4, 3],
        [3, 5]
    ]) + skel.max_node_id()
    # add tree
    skel.add_tree(
        nodes=nodes,
        edges=edges,
        name=tree_name)
    return skel
