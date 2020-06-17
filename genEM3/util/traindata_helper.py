import numpy as np

wkw_path = "/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1"
wkw_lims = np.asarray([19500, 15500, 0, 11000, 11000, 7000])

num_samples = 10
input_width = 302
mesh_width = 3
mesh_spacing = 10
sample_mesh_xy = np.arange(0,mesh_width*input_width, input_width)
sample_mesh_z = np.arange(0,999,10)


wkw_lims_valid = wkw_lims - np.asarray([0, 0, 0, 1000, 1000, 1000])
wkw_steps_valid = (wkw_lims_valid[3:6]/1000).astype(int)

rand_linds = np.random.randint(1, np.prod(wkw_steps_valid), num_samples)

