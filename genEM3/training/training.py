import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice


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
                 save=False
                 ):

        self.run_root = run_root
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.log_int = log_int
        self.device = torchDevice(device)

        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_root = os.path.join(run_root, '.log', time_str)
        self.save = save

        self.data_loaders = {"train": train_loader, "val": validation_loader}
        self.data_lengths = {"train": len(train_loader), "val": len(validation_loader)}

    def train(self):

        print('Starting training ...')

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        writer = SummaryWriter(self.log_root)
        self.model = self.model.to(self.device)

        it = 0
        for epoch in range(self.num_epoch):

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
                              format(epoch, i + 1, running_loss_avg))
                        writer.add_scalars('running_loss', {phase: running_loss}, it)
                        running_loss = 0.0

                epoch_loss_avg = epoch_loss / self.data_lengths[phase]
                print('Phase: ' + phase + ', epoch: {}: epoch loss: {:0.3f}'.
                      format(epoch, epoch_loss_avg))
                writer.add_scalars('epoch_loss', {phase: epoch_loss_avg}, epoch)
                writer.add_histogram('input histogram', inputs.cpu().data.numpy()[0, 0].flatten(), epoch)
                writer.add_histogram('output histogram', outputs.cpu().data.numpy()[0, 0].flatten(), epoch)
                writer.add_figure(
                    'images ' + phase, Trainer.show_imgs(inputs, outputs, list(range(0, inputs.shape[0], 2))), epoch)

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.log_root, 'torch_model'))

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
        fig, axs = plt.subplots(len(inds), 2)
        for i, idx in enumerate(inds):
            axs[i, 0].imshow(inputs[idx].data.numpy().squeeze(), cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(outputs[idx].data.numpy().squeeze(), cmap='gray')
            axs[i, 1].axis('off')

        return fig