""" This module contains the functionality related to filesystem directories"""
import os


def getAbsPathRepository():
    """ Return the absolute directory of the genEM3 to use for navigation"""
    curFileRoot = os.path.dirname(os.path.abspath(__file__))
    path_list = curFileRoot.split(os.sep)
    # by default index returns the index of the first occurance of genEM3
    genEM3_index = path_list.index('genEM3')
    # choose the elements of the repository main directory and join them
    path_repoElements = path_list[0:genEM3_index+1]
    return os.sep.join(path_repoElements)


def getDataDir():
    """return the path for the 'data' directory"""
    repoDir = getAbsPathRepository()
    return os.path.join(repoDir, 'data')


def getMag8DatasetDir():
    """return string for mag8 dataset we work with"""
    wkwDir = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'
    return wkwDir
