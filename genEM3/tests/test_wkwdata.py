""" This module contains the tests related to the WkwData module. In short, WkwData creates a
    pytorch dataset from a set of parameters. """
import os
import re
import pytest
from typing import List
import itertools

from genEM3.data.wkwdata import WkwData, DataSplit
import genEM3.util.path as gpath
from genEM3.data.transforms.normalize import ToZeroOneRange

def WkwDataSetConstructor():
    """ Construsts a WkwData[set] from fixed parameters. These parameters can also be explored for 
        further testing"""    
    # Get data source from example json
    json_dir = gpath.getDataDir()
    datasources_json_path = os.path.join(json_dir, 'datasource_20X_980_980_1000bboxes.json')
    data_sources = WkwData.datasources_from_json(datasources_json_path)
    # Only pick the first two bboxes for faster epoch
    data_sources = data_sources[0:2]
    data_split = DataSplit(train=0.70, validation=0.00, test=0.30)
    # input, output shape
    input_shape = (28, 28, 1)
    output_shape = (28, 28, 1)
    # flags for memory and storage caching
    cache_RAM = True
    cache_HDD = True
    # HDD cache directory
    connDataDir = '/conndata/alik/genEM3_runs/VAE/'
    cache_root = os.path.join(connDataDir, '.cache/')
    dataset = WkwData(
        input_shape=input_shape,
        target_shape=output_shape,
        data_sources=data_sources,
        data_split=data_split,
        normalize=False,
        transforms=ToZeroOneRange(minimum=0, maximum=255),
        cache_RAM=cache_RAM,
        cache_HDD=cache_HDD,
        cache_HDD_root=cache_root
    )
    return dataset


def getIndsPropString(wkwdata):
    """Get the string for the property containing the indices of train, validation and test data"""
    allAttributes = dir(wkwdata)
    r = re.compile('^data_.*_inds$')
    return list(filter(r.match, allAttributes))


def getIndices(wkwdata, indsPropStr: List[str]):
    """ Get the list of indices from the string of the dataset property"""
    return [getattr(wkwdata, indexString) for indexString in indsPropStr]

# The pytest fixture is run before every function that accepts it by default. We changed the scope
# to the module so that it would be only run once and all the tests in this module share the same 
# dataset
@pytest.fixture(scope='module')
def wkwdataset():
    return WkwDataSetConstructor()


def test_indexLengthCheck(wkwdataset):
    """Test whether the dataset length matches the length of train, validation and test datasets"""
    indexStrings = getIndsPropString(wkwdataset)
    indexList = getIndices(wkwdataset, indexStrings)
    lengthSeparate = [len(index) for index in indexList]
    assert sum(lengthSeparate) == len(wkwdataset)


def test_indexCoversRange(wkwdataset):
    """Test that the indices of the training data covers the full training set defined by json. This
        set is equivalent to a continous list of indices starting from 0 and ending in the length-1
    """
    indexStrings = getIndsPropString(wkwdataset)
    indexList = getIndices(wkwdataset, indexStrings)
    listOfIndices = list(itertools.chain.from_iterable(indexList))
    setOfIndices = set(itertools.chain.from_iterable(indexList))

    # Make sure the indices are unique by checking the length of the set (only keeps unique elements)
    # Note: The equality of the length of the list and set indices also indicates that there are no
    # overlaps between indices of the separate groups
    assert len(listOfIndices) == len(setOfIndices)

    # Also make sure that indices are continous (no edge cases are accidentally ignored)
    continousIndices = set(range(len(setOfIndices)))
    assert setOfIndices == continousIndices
