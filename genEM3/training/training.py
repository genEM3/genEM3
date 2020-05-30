import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice


class Trainer:

    def __init__(self,
                 run_root,
                 dataloader,
                 model,
                 optimizer,
                 criterion,
                 num_epoch,
                 log_int=10,
                 device='cpu'
                 ):

        self.run_root = run_root
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.log_int = log_int
        # convert the string to torch.device object
        self.device = torchDevice(device)

        self.log_root = os.path.join(run_root, '.log')

    def train(self):

        print('Starting training ...')

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        writer = SummaryWriter(self.log_root)
        # Copy model to specified device
        self.model = self.model.to(self.device)
        it = 0
        for epoch in range(self.num_epoch):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.dataloader):
                it += 1
                # copy input and targets to the device object
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i + 1) % self.log_int == 0:
                    print('it: {} (epoch: {}, batch: {}), running loss: {:0.3f}'.format(it, epoch, i + 1, running_loss))

                    writer.add_scalar('loss', loss.item(), it)
                    writer.add_scalar('running_loss', running_loss, it)
                    writer.add_figure('image', self.show_img(inputs, outputs, 0), it)
                    writer.add_figure('hist', self.show_hist(inputs, outputs, 0), it)

                    running_loss = 0.0

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
