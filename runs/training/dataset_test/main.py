import os
import time
from matplotlib import pyplot as plt
from genEM3.data.wkwdata import WkwData, DataSource

# Parameters
run_root = os.path.dirname(os.path.abspath(__file__))

wkw_root = '/gaba/tmpscratch/webknossos/Connectomics_Department/' \
                  '2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'

cache_root = os.path.join(run_root, '.cache/')
datasources_json_path = os.path.join(run_root,'datasources.json')
data_strata = {'training': [1, 2], 'validate': [3], 'test': []}
input_shape = (250, 250, 5)
output_shape = (125, 125, 3)

# Run
data_sources = WkwData.datasources_from_json(datasources_json_path)

# No Caching
dataset = WkwData(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    target_shape=output_shape,
    cache_root=None,
    cache_wipe=True,
    cache_size=1024,#MiB
    cache_dim=2,
    cache_range=8
)

t0 = time.time()
for sample_idx in range(8):
    print(sample_idx)
    data = dataset.get_ordered_sample(sample_idx)
    plt.imshow(data[0][0,:,:,0].data.numpy())
t1 = time.time()
print('No caching: {} seconds'.format(t1-t0))


# With Caching (cache empty)
dataset = WkwData(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    target_shape=output_shape,
    cache_root=cache_root,
    cache_wipe=True,
    cache_size=1024,#MiB
    cache_dim=2,
    cache_range=8
)

t0 = time.time()
for sample_idx in range(8):
    print(sample_idx)
    data = dataset.get_ordered_sample(sample_idx)
    plt.imshow(data[0][0,:,:,0].data.numpy())
t1 = time.time()
print('Empty cache: {} seconds'.format(t1-t0))


# With Caching (cache filled)
dataset = WkwData(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    target_shape=output_shape,
    cache_root=cache_root,
    cache_wipe=False,
    cache_size=1024,#MiB
    cache_dim=2,
    cache_range=8
)

t0 = time.time()
for sample_idx in range(8):
    print(sample_idx)
    data = dataset.get_ordered_sample(sample_idx)
    plt.imshow(data[0][0,:,:,0].data.numpy())
t1 = time.time()
print('Filled cache: {} seconds'.format(t1-t0))
