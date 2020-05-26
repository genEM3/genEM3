import pandas as pd
from genEM3.data.wkwdata import Wkwdata, DataSource

wkw_root_remote = '/gaba/tmpscratch/webknossos/Connectomics_Department/' \
                  '2018-11-13_scMS109_1to7199_v01_l4_06_24_fixed_mag8/color/1'
wkw_root_local = '../../.data/scMS109_1to7199_v01_subset/8-8-1/'

wkw_root = wkw_root_local

datasource_1 = DataSource(
    input_path=wkw_root,
    input_dtype='uint8',
    input_bbox=[20001, 16001, 0, 500, 500, 50],
    label_path=wkw_root,
    label_dtype='uint8',
    label_bbox=[20001, 16001, 0, 500, 500, 50]
    )

datasource_2 = DataSource(
    input_path=wkw_root,
    input_dtype='uint8',
    input_bbox=[20501, 16501, 0, 500, 500, 50],
    label_path=wkw_root,
    label_dtype='uint8',
    label_bbox=[20501, 16501, 0, 500, 500, 50]
    )

datasource_3 = DataSource(
    input_path=wkw_root,
    input_dtype='uint8',
    input_bbox=[21001, 17001, 0, 500, 500, 50],
    label_path=wkw_root,
    label_dtype='uint8',
    label_bbox=[20501, 16501, 0, 500, 500, 50]
    )

data_sources = [datasource_1, datasource_2, datasource_3]
data_strata = {'train': [1, 2], 'validate': [3], 'test': []}

input_shape = (250, 250, 100)
output_shape = (125, 125, 50)

dataset = Wkwdata(
    data_sources=data_sources,
    data_strata=data_strata,
    input_shape=input_shape,
    output_shape=output_shape,
)


