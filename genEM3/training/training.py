import os
import matplotlib.pyplot as plt
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
        self.log_root = os.path.join(run_root, '.log')
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

                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                running_loss = 0.0
                for i, (inputs, targets) in enumerate(self.data_loaders[phase]):
                    it += 1
                    # copy input and targets to the device object
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
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
                    if (i + 1) % self.log_int == 0:
                        print('it: {} (epoch: {}, batch: {}), running loss: {:0.3f}'.format(it, epoch, i + 1,
                                                                                            running_loss))

                        writer.add_scalar('loss_'+phase, loss.item(), it)
                        writer.add_scalar('running_loss_'+phase, running_loss, it)
                        writer.add_figure('image'+phase, self.show_img(inputs, outputs, 0), it)
                        writer.add_figure('hist'+phase, self.show_hist(inputs, outputs, 0), it)

                        running_loss = 0.0

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.run_root, 'torch_model'))

    @staticmethod
    def show_img(inputs, outputs, idx):
        curInput, curOutput = Trainer.copy2cpu(inputs, outputs, idx)
        # plot the input and output images as subplots
        fig, axs = plt.subplots(1, 2, figsize=(4, 3))
        img_input = curInput.data.numpy().squeeze()
        axs[0].imshow(img_input, cmap='gray')
        img_output = curOutput.data.numpy().squeeze()
        axs[1].imshow(img_output, cmap='gray')

        return fig

    @staticmethod
    def show_hist(inputs, outputs, idx):

        curInput, curOutput = Trainer.copy2cpu(inputs, outputs, idx)
        fig, axs = plt.subplots(1, 2, figsize=(4, 3))
        axs[0].hist(curInput.data.numpy().flatten())
        axs[1].hist(curOutput.data.numpy().flatten())

        return fig

    @staticmethod
    def copy2cpu(inputs, outputs, idx):
        # copies the specified training example (idx) to the cpu
        if inputs.is_cuda or outputs.is_cuda:
            curInput_cpu = inputs[idx].cpu()
            curOutput_cpu = outputs[idx].cpu()
        else:
            curInput_cpu = inputs[idx]
            curOutput_cpu = outputs[idx]
        return curInput_cpu, curOutput_cpu
