from serpentyne.data.convert import mat2hdf5
from serpentyne.data.dataset import D3d
from serpentyne.data.normalize import Normalizer
import glob
import os

data_source_dir = "/home/drawitschf/Code/mpi/florian/myelinPrediction/data/"
input_shape = [201, 201, 51]
output_shape = [151, 151, 51]
data_format = 'mat'
data_include_idx = [0,1,2]
pad_target = True

mat_filenames = glob.glob(os.path.join(data_source_dir, '*.' + data_format))
mat_input_name = 'raw'
mat_target_name = 'seg'
hdf5_filename = '/home/drawitschf/Code/mpi/florian/myelinPrediction/data/ex145_myelin.hdf5'
hdf5_input_name = 'input'
hdf5_target_name = 'target'

mat2hdf5(mat_filenames,
         mat_input_name,
         mat_target_name,
         hdf5_filename,
         hdf5_input_name,
         hdf5_target_name)

path_source = '/home/drawitschf/Code/mpi/florian/myelinPrediction/data/ex145_myelin.hdf5'

normalizer = Normalizer(path_source=path_source, in_place=True)

normalizer.get_stats('target')
print(normalizer.stats['summary'])
normalizer.get_stats('input')
print(normalizer.stats['summary'])

normalizer.normalize()

normalizer.get_stats('target')
print(normalizer.stats['summary'])
normalizer.get_stats()
print(normalizer.stats['summary'])
