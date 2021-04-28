import os

from genEM3.data.wkwdata import WkwData
from genEM3.util.path import get_data_dir

# Read the two jsons
target_names = ['Debris', 'Myelin']
json_names = ['combined_20K_patches.json', 'combined_20K_patches_v2.json']
full_names = [os.path.join(get_data_dir(), 'combined', f_name) for f_name in json_names]
ds_list = [WkwData.read_short_ds_json(name) for name in full_names]
ds_dict = [WkwData.convert_ds_to_dict(ds) for ds in ds_list]
# Get the difference between the two data sources from jsons 
diff_sources = WkwData.compare_ds_targets(two_datasources=ds_dict,
                                          source_names=json_names,
                                          target_names=target_names)
print(diff_sources)
