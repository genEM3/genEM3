import numpy as np
from genEM3.data.wkwdata import WkwData, DataSource

wkw_path = "/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1"
wkw_lims = np.asarray([21500, 15500, 0, 8000, 11000, 6000])

num_samples = 20

sample_dims = (906, 906, 1000)

input_mean = 148.0
input_std = 36.0

input_path = wkw_path
target_path = wkw_path

json_path = 'datasources_distributed.json'

sample_pos_x = np.random.randint(wkw_lims[0], wkw_lims[0]+wkw_lims[3]-sample_dims[0], num_samples)
sample_pos_y = np.random.randint(wkw_lims[1], wkw_lims[1]+wkw_lims[4]-sample_dims[1], num_samples)
sample_pos_z = np.random.randint(wkw_lims[2], wkw_lims[2]+wkw_lims[5]-sample_dims[2], num_samples)

datasources = []
for id in range(num_samples):
    input_bbox = [int(sample_pos_x[id]), int(sample_pos_y[id]), int(sample_pos_z[id]),
                  sample_dims[0], sample_dims[1], sample_dims[2]]
    target_bbox = input_bbox
    datasource = DataSource(id, input_path, input_bbox, input_mean, input_std, target_path, target_bbox)
    datasources.append(datasource)

WkwData.datasources_to_json(datasources, json_path)
