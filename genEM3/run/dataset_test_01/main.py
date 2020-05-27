import os
import time
from matplotlib import pyplot as plt
from genEM3.data.wkwdata import WkwData, DataSource

run_root = os.path.dirname(os.path.abspath(__file__))
wkw_root = '/gaba/tmpscratch/webknossos/Connectomics_Department/' \
                  '2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'

cache_root = os.path.join(run_root, '.cache/')
if not os.path.exists(cache_root):
    os.makedirs(cache_root)

datasource_1 = DataSource(
    id=1,
    input_path=wkw_root,
    input_dtype='uint8',
    input_bbox=[20001, 16001, 0, 200, 200, 2],
    target_path=wkw_root,
    target_dtype='uint8',
    target_bbox=[20001, 16001, 0, 200, 200, 2]
    )

datasource_2 = DataSource(
    id=2,
    input_path=wkw_root,
    input_dtype='uint8',
    input_bbox=[20501, 16501, 0, 200, 200, 2],
    target_path=wkw_root,
    target_dtype='uint8',
    target_bbox=[20501, 16501, 0, 200, 200, 2]
    )

datasource_3 = DataSource(
    id=3,
    input_path=wkw_root,
    input_dtype='uint8',
    input_bbox=[21001, 17001, 0, 500, 500, 50],
    target_path=wkw_root,
    target_dtype='uint8',
    target_bbox=[20501, 16501, 0, 500, 500, 50]
    )

data_sources = [datasource_1, datasource_2, datasource_3]
WkwData.datasources_to_json(data_sources, os.path.join(run_root, 'datasources.json'))
data_strata = {'train': [1, 2], 'validate': [3], 'test': []}

input_shape = (250, 250, 5)
output_shape = (125, 125, 3)

dataset = WkwData(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    output_shape=output_shape,
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

dataset = WkwData(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    output_shape=output_shape,
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

dataset = WkwData(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    output_shape=output_shape,
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
