import pandas as pd
from gen-EM.data.wkwdata import WkwDataset, DataSource

datasource_1 = DataSource(
    type='train',
    input_path='../../.data/scMS109_1to7199_v01_subset/8-8-1/',
    input_dtype='uint8',
    input_bbox=[20000, 16000, 0, 500, 500, 500],
    label_path='../../.data/scMS109_1to7199_v01_subset/8-8-1/',
    label_dtype='uint8',
    label_bbox=[20000, 16000, 0, 500, 500, 500]
    )

datasource_2 = DataSource(
    type='train',
    input_path='../../.data/scMS109_1to7199_v01_subset/8-8-1/',
    input_dtype='uint8',
    input_bbox=[20501, 16501, 0, 500, 500, 50],
    label_path='../../.data/scMS109_1to7199_v01_subset/8-8-1/',
    label_dtype='uint8',
    label_bbox=[20501, 16501, 0, 500, 500, 50]
    )

data_sources = [datasource_1, datasource_2]
input_shape = (250, 250, 100)
output_shape = (125, 125, 50)

dataset = WkwDataset(
    data_sources=data_sources,
    input_shape=input_shape,
    output_shape=output_shape,
)


