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
# %%
# Loaded the json file for the dataset
json_dir = os.path.join(get_data_dir(), 'debris_clean_added_bboxes2_wiggle_datasource.json') 
config = WkwData.config_wkwdata(json_dir)
dataset = WkwData.init_from_config(config)
# %%
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
assert all(l == dataset_lengths[0] for l in dataset_lengths)
# break down the range into partitions of 1000
range_size = 1000
list_ranges = annotation.divide_range(total_size=len(dataset_dict['large']),
                                      chunk_size=range_size)
# %%
#Annotate data using pigeon
annotation_fun = lambda i: display_example(i, dataset=larger_dataset, margin=margin, roi_size=roi_size)
annotations = []
for cur_range in list_ranges:
    print(f'Following range is {cur_range}')
    cur_a = annotate(cur_range,
                     options=['clean', 'debris', 'myelin'],
                     display_fn=annotation_fun)
    annotations.append(cur_a)


# %%
#save annotations
fname = 'annotationList_original_v04.pkl'
with open(fname, 'wb') as fp:
    pickle.dump(annotations, fp)

# Test reading it backabs
with open (fname, 'rb') as fp:
    annotations_reloaded = pickle.load(fp)
    
assert annotations == annotations_reloaded
print(annotations_reloaded[8][-1])


# %%
# Create one list with the concatenation of individual batches of ~1000
annotations_list = list(itertools.chain.from_iterable(annotations))
# Check that the indices in the list are a continuous range
assert [a for (a, _) in annotations_list ] == list(range(len(annotations_list)))


# %%
# Convert the annotations to numbers
types = ['clean', 'debris', 'myelin']
name_to_target = {'clean': 0.0, 'debris': 1.0, 'myelin': 0.0}

index_target_tuples = [(a[0], name_to_target[a[1]]) for a in annotations_list]
source_list = update_data_source_targets(patch_dataset, index_target_tuples)


# %%
print(source_list[-1])
print(patch_dataset.data_sources[-1])


# %%
# Json name
json_name = os.path.join(get_data_dir(), 'debris_clean_added_bboxes2_wiggle_datasource_without_myelin_v01.json')
# Write to json file
WkwData.datasources_to_json(source_list, json_name)


# %%
# Merge json files to get the training data
from genEM3.data.annotation import merge_json_from_data_dir
fnames_with = ['original_dataset_8562_patches/debris_clean_added_bboxes2_wiggle_datasource_without_myelin_v01.json', 
               'dense_3X_10_10_2_um/test_data_three_bboxes_without_myelin_v01.json']
output = 'dense_3X_10_10_2_um/original_merged_without_myelin_v01.json'
all_data_sources = merge_json_from_data_dir(fnames=fnames_with, output_fname=output)


# %%
# Look at individual examples
test_target = [(i,t) for (i,t) in index_target_tuples if int(t)==1]
for (i,t) in test_target:
    print(f'sample index: {i}, AK: {types[int(t)]}')
    annotation_fun(i)


# %%
# compare targets annotated by me and Flo
name_to_target = {'clean': 0, 'debris': 1, 'myelin': 0}
targets_AK = [name_to_target[a[1]] for a in annotations]
targets_Flo = [int(dataset.get_target_from_sample_idx(i)) for i in range(len(dataset))]

list_tuples = list(zip(targets_AK, targets_Flo))

agreement_list = [int(l[0] == l[1]) for l in list_tuples]

print(f'The number of disagreements: {len(agreement_list) - sum(agreement_list)}')


# %%
# Find the disagreements
index_disagreement = [i for i, cond in enumerate(agreement_list) if not cond]


# %%
# write the disagreements to an NML file
from genEM3.data.skeleton import add_bbox_tree
from wkskel import Skeleton
from genEM3.util.path import get_runs_dir

types = ['clean', 'debris']
skel = Skeleton(os.path.join(get_runs_dir(), 'inference/ae_classify_11_parallel/empty.nml'))
input_shape = (140, 140, 1)
# Write to nml
for i in index_disagreement:
    tree_name = f'sample index: {i}, AK: {types[targets_AK[i]]}, Flo: {types[targets_Flo[i]]}, your opinion:'
    sample_center = dataset.get_center_for_sample_idx(i)
    add_bbox_tree(sample_center, input_shape, tree_name, skel)
    
skel.write_nml(os.path.join(get_data_dir(), 'test_dataset_annotation_disagreement_v01.nml'))