from genEM3.util.path import getDataDir
from genEM3.data.wkwdata import WkwData
import os

# Test concatenating jsons
test_json_path = os.path.join(getDataDir(), 'test_data_three_bboxes.json') 
train_json_path = os.path.join(getDataDir(), 'debris_clean_added_bboxes2_wiggle_datasource.json')
# Concatenate the test and training data sets
output_name = os.path.join
all_ds = WkwData.concat_datasources([train_json_path, test_json_path], os.path.join(getDataDir(), 'train_test_combined.json'))
assert len(all_ds) == len(WkwData.datasources_from_json(test_json_path))+len(WkwData.datasources_from_json(train_json_path))
