import os
import cProfile
import pstats

from genEM3.util.path import get_data_dir
from genEM3.data.skeleton import make_skel_from_json

# start profiling
profiler = cProfile.Profile()
profiler.enable()

# read the json data sources
json_path = os.path.join(get_data_dir(), 'combined', 'combined_20K_patches.json')
skel = make_skel_from_json(json_path=json_path)

# Finish profiling
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
# write the nml file
nml_name = os.path.join(get_data_dir(), 'NML', 'combined_20K_patches.nml')
skel.write_nml(nml_write_path=nml_name)
