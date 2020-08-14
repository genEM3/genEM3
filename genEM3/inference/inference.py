import os
import time
import datetime
from typing import Optional, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import device as torchDevice
from genEM3.util import gpu, viewData
from genEM3.inference.writer import DataWriter

class Predictor:

    def __init__(self,
                 dataloader: DataLoader,
                 datawriters: Sequence[DataWriter],
                 model: torch.nn.Module,
                 state_dict: Optional[dict] = None,
                 device: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 output_shape: Optional[Tuple[int, ...]] = None,
                 interpolate: str = None):

        self.dataloader = dataloader
        self.datawriters = datawriters
        self.model = model
        self.state_dict = state_dict
        self.device = device
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.out_shape = output_shape
        self.interpolate = interpolate

    @torch.no_grad()
    def predict(self):
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Starting Inference ... ')
        start_time = time.time()

        for batch_idx, (data, sample_inds) in enumerate(self.dataloader):
            print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Predicting batch {}/{} ... '
                  .format(batch_idx, len(self.dataloader)))

            inputs = data['input']
            outputs = self.model(inputs)

            # print(sample_inds)
            # for i, sample_idx in enumerate(sample_inds):
            #     source_idx, bbox = self.dataloader.dataset.get_bbox_for_sample_idx(sample_idx)
            #     patch_input = inputs.detach().numpy()[i, 0, :, :]
            #     patch_output = np.exp(outputs.detach().numpy())[i, 1, :, :]
            #     fig, axs = plt.subplots(1, 3)
            #     axs[0].imshow(np.flipud(np.rot90(patch_input)), cmap='gray')
            #     axs[1].imshow(patch_output, cmap='Reds', vmin=0, vmax=1)
            #     axs[1].text(0, 0, '{:1.2f}'.format(float(patch_output)))
            #     axs[2].imshow(np.asarray(0 if float(patch_output) < 0.5 else 1).reshape(1, 1), cmap='gray', vmin=0, vmax=1)
            #     path_tmp = '/home/drawitschf/Code/genEM3/runs/inference/ae_classify_03_simple/tiff'
            #     fname_tmp = 'source{}-bbox{}_{}_{}_{}_{}_{}.tiff'.format(source_idx, *bbox)
            #     plt.savefig(os.path.join(path_tmp, fname_tmp))
            #     plt.close(fig)


            for datawriter in self.datawriters.values():
                datawriter.batch_to_cache(outputs, sample_inds)

        elapsed_time = time.time() - start_time
        print(elapsed_time)

        for datawriter in self.datawriters.values():
            if self.interpolate is not None:
                datawriter.interpolate_sparse_cache(method=self.interpolate)

            datawriter.cache_to_wkw(output_wkw_root=datawriter.output_path)

