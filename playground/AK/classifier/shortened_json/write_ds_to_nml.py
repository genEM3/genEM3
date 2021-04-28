import os
import wkskel

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

skel = add_bbox_tree(skel=empty_skel,
                     bbox=data_sources_dict['datasource_0']['input_bbox'],
                     tree_name='datasource_0')

skel.write_nml('test.nml')