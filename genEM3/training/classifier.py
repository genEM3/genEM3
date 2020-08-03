import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice
from genEM3.util import gpu


class Trainer:

    def __init__(self,
                 run_root,
                 model,
                 optimizer,
                 criterion,
                 train_loader,
                 validation_loader=None,
                 num_epoch=100,
                 log_int=10,
                 device='cpu',
                 save=False,
                 resume=False,
                 gpu_id: int = None
                 ):

        self.run_root = run_root
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.log_int = log_int
        self.save = save
        self.resume = resume

        if device == 'cuda':
            gpu.get_gpu(gpu_id)
            device = torch.device(torch.cuda.current_device())
        
        self.device = torchDevice(device)
        self.log_root = os.path.join(run_root, '.log')
        self.data_loaders = {"train": train_loader, "val": validation_loader}
        self.data_lengths = {"train": len(train_loader), "val": len(validation_loader)}

        if save:
            if not os.path.exists(self.log_root):
                os.makedirs(self.log_root)

    def train(self):

        if self.resume:
            print('Resuming training ...')
            checkpoint = torch.load(os.path.join(self.log_root, 'torch_model'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('Starting training ...')

        writer = SummaryWriter(self.log_root)
        self.model = self.model.to(self.device)

        epoch = int(self.model.epoch) + 1
        it = int(self.model.iteration)
        for epoch in range(epoch, epoch + self.num_epoch):

            epoch_root = 'epoch_{:02d}'.format(epoch)
            if not os.path.exists(os.path.join(self.log_root, epoch_root)):
                os.makedirs(os.path.join(self.log_root, epoch_root))

            for phase in ['train', 'val']:
                epoch_loss = 0

                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                running_loss = 0.0
                for i, data in enumerate(self.data_loaders[phase]):
                    it += 1
                    # copy input and targets to the device object
                    inputs = data['input'].to(self.device)
                    targets = data['target'].to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    if (i + 1) % self.log_int == 0:
                        running_loss_avg = running_loss/self.log_int
                        print('Phase: ' + phase + ', epoch: {}, batch {}: running loss: {:0.3f}'.
                              format(self.model.epoch, i + 1, running_loss_avg))
                        writer.add_scalars('running_loss', {phase: running_loss_avg}, it)
                        running_loss = 0.0

                epoch_loss_avg = epoch_loss / self.data_lengths[phase]
                print('Phase: ' + phase + ', epoch: {}: epoch loss: {:0.3f}'.
                      format(epoch, epoch_loss_avg))
                writer.add_scalars('epoch_loss', {phase: epoch_loss_avg}, epoch)
                writer.add_histogram('input histogram', inputs.cpu().data.numpy()[0, 0].flatten(), epoch)
                writer.add_histogram('output histogram', outputs.cpu().data.numpy()[0, 0].flatten(), epoch)
                figure_inds = list(range(inputs.shape[0]))
                figure_inds = figure_inds if len(figure_inds) < 4 else list(range(4))
                fig = Trainer.show_imgs(inputs, outputs, figure_inds)
                fig.savefig(os.path.join(self.log_root, epoch_root, phase+'.png'))
                writer.add_figure(
                    'images ' + phase, fig, epoch)

                if self.save & (phase == 'train'):
                    print('Saving model state...')

                    self.model.epoch = torch.nn.Parameter(torch.tensor(epoch), requires_grad=False)
                    self.model.iteration = torch.nn.Parameter(torch.tensor(it), requires_grad=False)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, os.path.join(self.log_root, epoch_root, 'model_state_dict'))
                    torch.save({
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.log_root, 'optimizer_state_dict'))

        print('Finished training ...')

        # dictionary of accuracy metrics for tune hyperparameter optimization
        return {"val_loss_avg": epoch_loss_avg}

    @staticmethod
    def copy2cpu(inputs, outputs):
        if inputs.is_cuda:
            inputs = inputs.cpu()
        if outputs.is_cuda:
            outputs = outputs.cpu()
        return inputs, outputs

    @staticmethod
    def n1hw_to_n3hw(data):
        return data.cpu().repeat(1, 3, 1, 1)

    @staticmethod
    def show_img(inputs, outputs, idx):
        inputs, outputs = Trainer.copy2cpu(inputs, outputs)
        fig, axs = plt.subplots(1, 2, figsize=(4, 3))
        axs[0].imshow(inputs[idx].data.numpy().squeeze(), cmap='gray')
        axs[1].imshow(outputs[idx].data.numpy().squeeze(), cmap='gray')
        return fig

    @staticmethod
    def show_imgs(inputs, outputs, inds):
        inputs, outputs = Trainer.copy2cpu(inputs, outputs)
        fig, axs = plt.subplots(1, len(inds), figsize=(3*len(inds), 6))
        for i, idx in enumerate(inds):
            input_ = inputs[idx].data.numpy().squeeze()
            output = outputs[idx].data.numpy().squeeze()
            if input_.shape != output.shape:
                output = np.tile(output, input_.shape)
            input_output = np.concatenate((input_, output), axis=0)
            axs[i].imshow(input_output, cmap='gray')
            axs[i].axis('off')
        plt.tight_layout()

        return fig
