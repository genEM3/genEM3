import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice
from genEM3.util import gpu
from genEM3.training.metrics import Metrics


class Trainer:

    def __init__(self,
                 run_name: str,
                 run_root: str,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.MSELoss,
                 data_loaders: {},
                 num_epoch: int = 100,
                 log_int: int = 10,
                 device: str = 'cpu',
                 save: bool = False,
                 save_int: int = 1,
                 resume: bool = False,
                 gpu_id: int = None
                 ):

        self.run_name = run_name
        self.run_root = run_root
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.log_int = log_int
        self.save = save
        self.save_int = save_int
        self.resume = resume

        if device == 'cuda':
            gpu.get_gpu(gpu_id)
            device = torch.device(torch.cuda.current_device())
        
        self.device = torchDevice(device)
        self.log_root = os.path.join(run_root, '.log', run_name)
        self.data_loaders = data_loaders
        self.data_lengths = dict(zip(self.data_loaders.keys(), [len(loader) for loader in self.data_loaders]))

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
        sample_inds = dict()
        for epoch in range(epoch, epoch + self.num_epoch):

            epoch_root = 'epoch_{:02d}'.format(epoch)
            if not os.path.exists(os.path.join(self.log_root, epoch_root)):
                os.makedirs(os.path.join(self.log_root, epoch_root))

            for phase in self.data_loaders.keys():

                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                epoch_loss = 0
                running_loss = 0.0
                target_sum = 0
                predicted_sum = 0
                correct_sum = 0
                batch_idx_start = 0

                num_items = len(self.data_loaders[phase].batch_sampler.sampler.indices)

                inputs_phase = -np.ones((num_items, 1, 140, 140)).astype(float)
                outputs_phase = -np.ones((num_items, 2)).astype(float)
                predictions_phase = -np.ones(num_items).astype(int)
                targets_phase = -np.ones(num_items).astype(int)
                correct_phase = -np.ones(num_items).astype(int)

                sample_ind_phase = []
                for i, data in enumerate(self.data_loaders[phase]):

                    it += 1

                    # copy input and targets to the device object
                    inputs = data['input'].to(self.device)
                    targets = data['target'].to(self.device)
                    sample_ind_batch = data['sample_idx']
                    sample_ind_phase.extend(sample_ind_batch)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    inputs, outputs, targets = Trainer.copy2cpu(inputs, outputs, targets)

                    predicted_classes = np.argmax(np.exp(outputs.detach().numpy()), axis=1)
                    predicted_sum += np.sum(predicted_classes)
                    target_classes = targets.detach().numpy()
                    target_sum += np.sum(target_classes)
                    correct_classes = predicted_classes == target_classes
                    correct_sum += np.sum(correct_classes)

                    if i > 0:
                        batch_idx_start = batch_idx_end
                    batch_idx_end = batch_idx_start + len(targets)
                    inputs_phase[batch_idx_start:batch_idx_end, :, :, :] = inputs.detach().numpy()
                    outputs_phase[batch_idx_start:batch_idx_end, :] = outputs.detach().numpy()
                    predictions_phase[batch_idx_start:batch_idx_end] = predicted_classes
                    targets_phase[batch_idx_start:batch_idx_end] = target_classes
                    correct_phase[batch_idx_start:batch_idx_end] = correct_classes

                    running_loss += loss.item()
                    epoch_loss += loss.item()

                    if i % self.log_int == 0:
                        running_loss_log = float(running_loss) / batch_idx_end
                        running_accuracy_log = float(correct_sum) / batch_idx_end
                        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')' + ' Phase: ' + phase +
                              ', epoch: {}, batch: {}, running loss: {:0.4f}, running accuracy: {:0.3f} '.
                              format(epoch, i, running_loss_log, running_accuracy_log))
                        writer.add_scalars('running_loss', {phase: running_loss_log}, it)
                        writer.add_scalars('running_accuracy', {phase: running_accuracy_log}, it)


                epoch_loss_log = float(epoch_loss) / num_items
                epoch_accuracy_log = float(correct_sum) / num_items
                print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')' + ' Phase: ' + phase +
                      ', epoch: {}: epoch loss: {:0.4f}, epoch accuracy: {:0.3f} '.
                      format(epoch, epoch_loss_log, epoch_accuracy_log))

                metrics = Metrics(
                    targets=targets_phase, outputs=outputs_phase, output_prob_fn=lambda x: np.exp(x[:, 1]),
                    sample_ind=sample_ind_phase)
                metrics.confusion_table(
                    path_out=os.path.join(self.log_root, epoch_root, 'confusion_table_' + phase + '.csv'))
                metrics.prediction_table(
                    path_out=os.path.join(self.log_root, epoch_root, 'prediction_table_' + phase + '.csv'))

                writer.add_scalars('epoch_loss', {phase: epoch_loss_log}, epoch)
                writer.add_scalars('epoch_accuracy', {phase: epoch_accuracy_log}, epoch)
                writer.add_scalars('precision/PPV', {phase: metrics.metrics['PPV']}, epoch)
                writer.add_scalars('recall/TPR', {phase: metrics.metrics['TPR']}, epoch)

                fig = Trainer.show_imgs(inputs=inputs_phase, outputs=outputs_phase, predictions=predictions_phase,
                                        targets=targets_phase,
                                        sample_ind=sample_ind_phase)
                figname = 'image_examples_'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'))
                writer.add_figure(figname + phase, fig, epoch)

                fig = Trainer.show_classification_matrix(targets=targets_phase, predictions=predictions_phase,
                                                         metrics=metrics.metrics)
                figname = 'targets_outputs_correct_'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'))
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.eps'))
                writer.add_figure(figname + phase, fig, epoch)

                writer.add_pr_curve(
                    'pr_curve_'+phase, labels=targets_phase, predictions=np.exp(outputs_phase[:, 1]), global_step=epoch,
                    num_thresholds=50)

                if self.save & (phase == 'train') & (epoch % self.save_int == 0):
                    print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Writing model graph ... ')
                    # writer.add_graph(self.model, inputs)

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

        writer.close()
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Closed writer ... ')

    @staticmethod
    def copy2cpu(inputs, outputs, targets):
        if inputs.is_cuda:
            inputs = inputs.cpu()
        if outputs.is_cuda:
            outputs = outputs.cpu()
        if targets.is_cuda:
            targets = targets.cpu()
        return inputs, outputs, targets

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
    def show_imgs(inputs, outputs, predictions, targets, sample_ind, class_idx=1, plot_ind = None):

        if plot_ind is None:
            plot_ind = list(range(5))

        fig, axs = plt.subplots(2, len(plot_ind), figsize=(3 * len(plot_ind), 6))
        for i, sample_idx in enumerate(np.asarray(sample_ind)[plot_ind]):
            input = inputs[i, 0, :, :].squeeze()
            axs[0, i].imshow(input, cmap='gray')
            axs[0, i].axis('off')
            output = np.tile(np.exp((outputs[i][class_idx])), (int(input.shape[0]/3), int(input.shape[1])))
            prediction = np.tile(int(predictions[i]), (int(input.shape[0]/3), int(input.shape[1])))
            target = np.tile(int(targets[i]), (int(input.shape[0]/3), int(input.shape[1])))
            fused = np.concatenate((output, prediction, target), axis=0)
            axs[1, i].imshow(fused, cmap='gray', vmin=0, vmax=1)
            axs[1, i].text(0.5, 0.875, 'sample_idx: {}'.format(sample_idx),
                           transform=axs[1, i].transAxes, ha='center', va='center', c=[0.8, 0.8, 0.2])
            axs[1, i].text(0.5, 0.75, 'output (class {:d}): {:01.2f}'.format(class_idx, np.exp((outputs[i][class_idx]))),
                             transform=axs[1, i].transAxes, ha='center', va='center', c=[0.8, 0.8, 0.2])
            axs[1, i].text(0.5, 0.5, 'prediction class: {:d}'.format(int(predictions[i])),
                             transform=axs[1, i].transAxes, ha='center', va='center', c=[0.5, 0.5, 0.5])
            axs[1, i].text(0.5, 0.2, 'target class: {:d}'.format(int(targets[i])),
                             transform=axs[1, i].transAxes, ha='center', va='center', c=[0.5, 0.5, 0.5])
            axs[1, i].axis('off')
            axs[0, i].set_ylabel('sample_idx: {}'.format(sample_idx))

        plt.tight_layout()

        return fig

    @staticmethod
    def show_classification_matrix(targets, predictions, metrics):

        targets_pr = targets.copy().astype(int)
        predictions_pr = predictions.copy().astype(int)
        correct_pr = ((targets_pr == 0) == (predictions_pr == 0)) | ((targets_pr == 1) == (predictions_pr == 1)) + 2
        code_pr = targets_pr.copy()
        code_pr[metrics['TN_idx']] = 4
        code_pr[metrics['TP_idx']] = 5
        code_pr[predictions_pr > targets_pr] = 6
        code_pr[predictions_pr < targets_pr] = 7

        mat = np.stack((targets_pr, predictions_pr, correct_pr, code_pr), axis=0)

        colors = [[0.0, 0.0, 0.0, 1],
                  [1.0, 1.0, 1.0, 1],
                  [1.0, 0.0, 0.0, 1],
                  [0.0, 1.0, 0.0, 1],
                  [0.4, 0.1, 0.9, 1],
                  [0.3, 0.5, 0.9, 1],
                  [0.9, 0.4, 0.1, 1],
                  [0.9, 0.1, 0.5, 1]]
        cmap = ListedColormap(colors=colors)
        fig_width_mult = min(max([0.5, len(targets)/2000]), 3)
        fig, axs = plt.subplots(figsize=(12*fig_width_mult, 6))
        axs.matshow(mat, cmap=cmap, vmin=0, vmax=7)
        axs.set_yticks([0, 1, 2, 3])
        axs.set_yticklabels(['targets', 'outputs', 'accuracy', 'confusion'])
        axs.set_aspect(10)
        bbox = axs.get_position().bounds
        axs2 = plt.axes((bbox[0], 0.1, bbox[2], 0.2), sharex=axs)
        axs2.text(0.010, 1.00, 'target|output', c=(0.2, 0.2, 0.2), weight='bold', transform=axs2.transAxes)
        axs2.text(0.017, 0.75, 'artifact: {}'.format(metrics['P']), c=(0.2, 0.2, 0.2), backgroundcolor=colors[1], transform=axs2.transAxes)
        axs2.text(0.017, 0.50, 'no artifact: {}'.format(metrics['N']), c=(0.8, 0.8, 0.8), backgroundcolor=colors[0], transform=axs2.transAxes)
        axs2.text(0.27, 1.00, 'accuracy', c=(0.2, 0.2, 0.2), weight='bold', transform=axs2.transAxes)
        axs2.text(0.20, 0.75, 'frac correct:   {:03d}/{:03d}={:01.2f}'.format(metrics['TP'] +
                  metrics['TN'], len(targets), (metrics['TP'] +
                  metrics['TN'])/len(targets)), c=(0.2, 0.2, 0.2), backgroundcolor=colors[3],
                  transform=axs2.transAxes)
        axs2.text(0.20, 0.50, 'frac incorrect: {:03d}/{:03d}={:01.2f}'.format(metrics['FP'] +
                  metrics['FN'], len(targets), (metrics['FP'] +
                  metrics['FN'])/len(targets)), c=(0.2, 0.2, 0.2), backgroundcolor=colors[2],
                  transform=axs2.transAxes)
        axs2.text(0.60, 1.00, 'confusion', c=(0.2, 0.2, 0.2), weight='bold', transform=axs2.transAxes)
        axs2.text(0.50, 0.75, 'TP: {:01.0f}'.format(metrics['TP']).ljust(12), c=(0.8, 0.8, 0.8),
                  backgroundcolor=colors[5], transform=axs2.transAxes)
        axs2.text(0.60, 0.75, 'FP: {:01.0f}'.format(metrics['FP']).ljust(12), c=(0.2, 0.2, 0.2),
                  backgroundcolor=colors[6], transform=axs2.transAxes)
        axs2.text(0.50, 0.50, 'FN: {:01.0f}'.format(metrics['FN']).ljust(12), c=(0.2, 0.2, 0.2),
                  backgroundcolor=colors[7], transform=axs2.transAxes)
        axs2.text(0.60, 0.50, 'TN: {:01.0f}'.format(metrics['TN']).ljust(12), c=(0.8, 0.8, 0.8),
                  backgroundcolor=colors[4], transform=axs2.transAxes)
        axs2.text(0.70, 0.75, 'Precision: {:01.2f}'.format(metrics['PPV']).ljust(20), c=(0.2, 0.2, 0.2),
                  backgroundcolor=(0.7, 0.7, 0.7), transform=axs2.transAxes)
        axs2.text(0.70, 0.50, 'Recall: {:01.2f}'.format(metrics['TPR']).ljust(20), c=(0.2, 0.2, 0.2),
                  backgroundcolor=(0.7, 0.7, 0.7), transform=axs2.transAxes)
        axs2.axis('off')

        return fig




