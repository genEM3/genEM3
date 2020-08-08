import numpy as np
from scipy.interpolate import griddata
from scipy.special import logit, expit

class DataWriter:

    def __init__(self,
                 dataloader,
                 output_collate_fn,
                 output_label,
                 output_path,
                 output_dtype,
                 output_dtype_fn):

        self.dataloader = dataloader
        self.output_collate_fn = output_collate_fn
        self.output_label = output_label
        self.output_path = output_path
        self.output_dtype = output_dtype
        self.output_dtype_fn = output_dtype_fn

    def batch_to_cache(self, outputs, output_inds):
        self.dataloader.dataset.write_output_to_cache(self.output_collate_fn(outputs), output_inds, self.output_label)

    def interpolate_sparse_cache(self, method='nearest'):
        for wkw_path in self.dataloader.dataset.data_cache_output.keys():
            for wkw_bbox in self.dataloader.dataset.data_cache_output[wkw_path].keys():
                cache = self.dataloader.dataset.data_cache_output[wkw_path][wkw_bbox][self.output_label]
                for z in range(cache.shape[2]):
                    data = cache[:, :, z]
                    points = np.argwhere(~np.isnan(data))
                    values = data[points[:, 0], points[:, 1]]
                    grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
                    data_dense = griddata(points, values, (grid_x, grid_y), method=method)
                    cache[:, :, z] = data_dense

    def cache_to_wkw(self, output_wkw_root):
        for wkw_path in self.dataloader.dataset.data_cache_output.keys():
            for wkw_bbox in self.dataloader.dataset.data_cache_output[wkw_path].keys():
                self.dataloader.dataset.wkw_write_cached(wkw_path, wkw_bbox,
                                                         output_wkw_root=output_wkw_root,
                                                         output_label=self.output_label,
                                                         output_dtype=self.output_dtype,
                                                         output_dtype_fn=self.output_dtype_fn)