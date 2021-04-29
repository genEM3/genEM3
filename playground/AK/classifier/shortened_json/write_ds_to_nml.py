import os
import wkskel
from tqdm import tqdm

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir
from genEM3.data.skeleton import add_bbox_tree

# create an empty skel to add the bounding boxes 
empty_skel_name = os.path.join(get_data_dir(), 'NML', 'empty_skel.nml')
empty_skel = wkskel.Skeleton(nml_path=empty_skel_name)

# read the json data sources
json_path = os.path.join(get_data_dir(), 'combined', 'combined_20K_patches.json')
data_sources_list = WkwData.read_short_ds_json(json_path=json_path)
data_sources_dict = WkwData.convert_ds_to_dict(datasources=data_sources_list)
# Init with empty skeleton
skel = empty_skel
# Loop over data source patches
keys = list(data_sources_dict.keys())
for key in tqdm(keys, desc='Making bbox nml', total=len(keys)):
    # Encode the target in the tree name
    cur_target = data_sources_dict[key]['target_class']
    cur_name = f'{key}, Debris: {cur_target[0]}, Myelin: {cur_target[1]}'
    # add current tree
    nodes, edges = add_bbox_tree(skel=skel,
                                 bbox=data_sources_dict[key]['input_bbox'])
    # add tree
    skel.add_tree(
        nodes=nodes,
        edges=edges,
        name=cur_name)

# write the nml file
nml_name = os.path.join(get_data_dir(), 'NML', 'combined_20K_patches.nml')
skel.write_nml(nml_write_path=nml_name)
