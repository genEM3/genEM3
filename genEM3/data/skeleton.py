import numpy as np
# Functions for manipulating the skeleton object (These could be moved to wkskel repo if not already there)


def getAllTreeCoordinates(skel):
    """Get the coordinates of all the nodes in trees within a skeleton object"""
    # Get the coordinate of the nodes as numpy array
    coordsList = [skel.nodes[x].position.values for x in range(skel.num_trees())]
    # Concatenate into a single array of coordinates
    return np.vstack(coordsList)
