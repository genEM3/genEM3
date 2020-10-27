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
