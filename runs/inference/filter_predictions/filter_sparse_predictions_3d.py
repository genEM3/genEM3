import os
from genEM3.inference.prediction_modifier import SparsePrediction, FILTER_KERNELS

run_root = os.path.dirname(os.path.abspath(__file__))
wkw_bbox = [28672, 24576, 5120, 1024, 1024, 1024]
input_wkw_root = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8_artifact_pred/probs_sparse/1'
output_wkw_root = '/tmpscratch/webknossos/Connectomics_Department/2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8_artifact_pred/probs_sparse_filt/1'

sp = SparsePrediction(input_wkw_root=input_wkw_root, output_wkw_root=output_wkw_root)

filter_kernel = FILTER_KERNELS['3d_gauss_sandwich_9']
sp.filter_sparse_cube_3d(wkw_bbox=wkw_bbox, filter_kernel=filter_kernel, compress_output=True)