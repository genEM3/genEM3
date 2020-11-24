# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import time
import pickle
import itertools
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from genEM3.data.wkwdata import WkwData,DataSource
from genEM3.util.path import get_data_dir
import genEM3.data.annotation as annotation 
# %% Prepare for annotation
# Loaded the json file for the dataset
json_dir = os.path.join(get_data_dir(), 'debris_clean_added_bboxes2_wiggle_datasource.json') 
config = WkwData.config_wkwdata(json_dir)
dataset = WkwData.init_from_config(config)

# Get a set of data sources with the normal bounding boxes to create a patch wise detaset and a larger bounding box for annotation
margin = 35
roi_size = 140
source_dict = annotation.patch_source_list_from_dataset(dataset=dataset,
                                                        margin=margin,
                                                        roi_size=roi_size)
dataset_dict = dict.fromkeys(source_dict)

for key in source_dict:
    cur_source = source_dict[key]
    cur_patch_shape = tuple(cur_source[0].input_bbox[3:6])
    cur_config = WkwData.config_wkwdata(datasources_json_path=None,
                                        input_shape=cur_patch_shape,
                                        output_shape=cur_patch_shape)
    dataset_dict[key] = WkwData.init_from_config(cur_config, source_dict[key])
# assert larger and small datasets have the same length
dataset_lengths = [len(d) for d in dataset_dict.values()]
assert all(cur_L == dataset_lengths[0] for cur_L in dataset_lengths)
# break down the range into partitions of 1000
range_size = 1000
list_ranges = annotation.divide_range(total_size=len(dataset_dict['large']),
                                      chunk_size=range_size)
# %%
#Annotate data using pigeon
annotation_fun = lambda i: annotation.display_example(i, dataset=dataset_dict['large'], margin=margin, roi_size=roi_size)
annotations = []
for cur_range in list_ranges:
    print(f'Following range is {cur_range}')
    cur_a = annotation.annotate(cur_range,
                     options=['clean', 'debris', 'myelin'],
                     display_fn=annotation_fun)
    annotations.append(cur_a)