import os
from scipy import stats, ndimage
import numpy as np
from genEM3.data.wkwdata import WkwData

FILTER_KERNELS = {
    '3d_gauss_sandwich_9': np.array([
        [[-0.025, -0.05, -0.025],
         [-0.05, -0.2, -0.05],
         [-0.025, -0.05, -0.025]],
        [[0.125, 0.25, 0.125],
         [0.25, 0.5, 0.25],
         [0.125, 0.25, 0.125]],
        [[-0.025, -0.05, -0.025],
         [-0.05, -0.2, -0.05],
         [-0.025, -0.05, -0.025]]]
    ).swapaxes(0, 2)
}


class SparsePrediction:

    def __init__(self,
                 input_wkw_root,
                 output_wkw_root):

        assert os.path.exists(input_wkw_root), 'input wkw root does not exist or cannot be reached'

        self.input_wkw_root = input_wkw_root
        self.output_wkw_root = output_wkw_root

    def filter_sparse_cube_3d(self, wkw_bbox, filter_kernel, compress_output=False):

        data = WkwData.wkw_read(self.input_wkw_root, wkw_bbox).squeeze(axis=0)
        pred_inds_sparse = np.where(~np.isnan(data))
        pred_inds_dense = [stats.rankdata(pis, method='dense') - 1 for pis in pred_inds_sparse]
        data_dense = np.zeros((max(pred_inds_dense[0]+1), max(pred_inds_dense[1])+1, max(pred_inds_dense[2])+1),
                              dtype=np.float32)

        for i, (xd, yd, zd) in enumerate(zip(*pred_inds_dense)):
            xs, ys, zs = [pis[i] for pis in pred_inds_sparse]
            data_dense[xd, yd, zd] = data[xs, ys, zs]

        data_dense_conv = ndimage.filters.convolve(data_dense, weights=filter_kernel)
        data_dense_conv = data_dense_conv/data_dense_conv.max()
        data_dense_conv[data_dense_conv < 0] = 0

        for i, (xs, ys, zs) in enumerate(zip(*pred_inds_sparse)):
            xd, yd, zd = [pid[i] for pid in pred_inds_dense]
            data[xs, ys, zs] = data_dense_conv[xd, yd, zd]

        data = np.expand_dims(data, axis=0)
        self.wkw_create_write(data=data, wkw_root=self.output_wkw_root, wkw_bbox=wkw_bbox, compress=compress_output)

    @staticmethod
    def wkw_create_write(data, wkw_root, wkw_bbox, compress=False):

        if compress:
            wkw_block_type= 2
        else:
            wkw_block_type = 1

        if not os.path.exists(wkw_root):
            os.makedirs(wkw_root)

        if not os.path.exists(os.path.join(wkw_root, 'header.wkw')):
            WkwData.wkw_create(wkw_root, data.dtype, wkw_block_type)

        WkwData.wkw_write(wkw_root, wkw_bbox, data)