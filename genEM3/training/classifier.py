import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice
from genEM3.util import gpu

np.seterr(all='raise')

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
            print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Resuming training ... ')
            checkpoint = torch.load(os.path.join(self.log_root, 'torch_model'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Starting training ... ')

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
                running_loss = 0.0
                target_sum = 0
                predicted_sum = 0
                correct_sum = 0

                num_items = len(self.data_loaders[phase].dataset)
                batch_size = self.data_loaders['train'].batch_size

                outputs_phase = np.ones(num_items).dtype(int) * 2
                targets_phase = np.ones(num_items).dtype(int) * 2
                correct_phase = np.ones(num_items).dtype(int) * 2

                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                for i, data in enumerate(self.data_loaders[phase]):

                    it += 1

                    # copy input and targets to the device object
                    inputs = data['input'].to(self.device)
                    targets = data['target'].to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    predicted_classes = np.argmax(np.exp(outputs.detach().numpy()), axis=1)
                    predicted_sum += np.sum(predicted_classes)
                    target_classes = targets.detach().numpy()
                    target_sum += np.sum(target_classes)
                    correct_classes = predicted_classes == target_classes
                    correct_sum += np.sum(correct_classes)

                    batch_idx_start = i * batch_size
                    batch_idx_end = batch_idx_start + len(targets)
                    outputs_phase[batch_idx_start:batch_idx_end] = predicted_classes
                    targets_phase[batch_idx_start:batch_idx_end] = target_classes
                    correct_phase[batch_idx_start:batch_idx_end] = correct_classes

                    running_loss += loss.item()
                    epoch_loss += loss.item()

                    if i > 0 & (i % self.log_int == 0):
                        running_loss_log = float(running_loss) / batch_idx_end
                        running_accuracy_log = float(correct_sum) / batch_idx_end
                        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')' + ' Phase: ' + phase +
                              ', epoch: {}, batch: {}, running loss: {:0.3f}, running accuracy: {:0.2f} '.
                              format(self.model.epoch, i + 1, running_loss_log, running_accuracy_log))
                        writer.add_scalars('running_loss', {phase: running_loss_log}, it)
                        writer.add_scalars('running_accuracy', {phase: running_accuracy_log}, it)

                epoch_loss_log = float(epoch_loss) / num_items
                epoch_accuracy_log = float(correct_sum) / num_items
                print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')' + ' Phase: ' + phase +
                      ', epoch: {}: epoch loss: {:0.3f}, epoch accuracy: {:0.2f} '.
                      format(epoch, epoch_loss_log, epoch_accuracy_log))
                writer.add_scalars('epoch_loss', {phase: epoch_loss_log}, epoch)
                writer.add_scalars('epoch_accuracy', {phase: epoch_accuracy_log}, epoch)
                figure_inds = list(range(inputs.shape[0]))
                figure_inds = figure_inds if len(figure_inds) < 4 else list(range(4))
                fig = Trainer.show_imgs(inputs, outputs, figure_inds)
                figname = 'image_examples_'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'))
                writer.add_figure(figname + phase, fig, epoch)
                fig = Trainer.show_classification_matrix(targets_phase, outputs_phase, correct_phase)
                figname = 'targets_outputs_correct_'
                writer.add_figure(figname + phase, fig, epoch)
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'))

                if self.save & (phase == 'train'):
                    print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Saving model state... ')

                    self.model.epoch = torch.nn.Parameter(torch.tensor(epoch), requires_grad=False)
                    self.model.iteration = torch.nn.Parameter(torch.tensor(it), requires_grad=False)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, os.path.join(self.log_root, epoch_root, 'model_state_dict'))
                    torch.save({
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.log_root, 'optimizer_state_dict'))

        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Finished training ... ')

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
    def show_imgs(inputs, outputs, inds, class_idx=1):
        inputs, outputs = Trainer.copy2cpu(inputs, outputs)
        fig, axs = plt.subplots(2, len(inds), figsize=(3*len(inds), 6))
        for i, idx in enumerate(inds):
            input_ = inputs[idx].data.numpy().squeeze()
            axs[0, idx].imshow(input_, cmap='gray')
            axs[0, idx].axis('off')
            output = outputs[idx].data.numpy().squeeze()
            im_output = axs[1, idx].imshow(np.tile(np.exp(output[class_idx]), input_.shape),
                                                  cmap='gray', vmin=0, vmax=1)
            axs[1, idx].axis('off')
            fig.colorbar(im_output, ax=axs[1, idx], shrink=0.7)
            axs[1, idx].text(0.5, 0.5, '{:1.2f}'.format(np.exp(output[class_idx])), horizontalalignment='center',
                                    verticalalignment='center', transform=axs[1, idx].transAxes)

        plt.tight_layout()

        return fig

    @staticmethod
    def show_classification_matrix(targets, outputs, correct):

        mat = np.stack((targets, outputs, correct), axis=0)
        colors = np.asarray([[0.1, 0.1, 0.1, 1], [0.9, 0.9, 0.9, 1], [0.8, 0.1, 0.3, 1]])
        cmap = ListedColormap(colors=colors)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.matshow(mat, cmap='gray', vmin=0, vmax=2)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['targets', 'outputs', 'correct'])
        ax.set_aspect(3)
        plt.tight_layout()

        return fig

