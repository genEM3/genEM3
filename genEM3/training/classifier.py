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

                if epoch == 1:
                    sample_inds[phase] = self.data_loaders[phase].batch_sampler.sampler.indices

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

                for i, data in enumerate(self.data_loaders[phase]):

                    it += 1

                    # copy input and targets to the device object
                    inputs = data['input'].to(self.device)
                    targets = data['target'].to(self.device)
                    index = data['sample_idx']
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

                metrics = Trainer.compute_metrics(targets=targets_phase, predictions=predictions_phase)

                writer.add_scalars('epoch_loss', {phase: epoch_loss_log}, epoch)
                writer.add_scalars('epoch_accuracy', {phase: epoch_accuracy_log}, epoch)
                writer.add_scalars('precision', {phase: metrics['precision']}, epoch)
                writer.add_scalars('recall', {phase: metrics['recall']}, epoch)

                sample_inds_epoch = [
                    self.data_loaders[phase].batch_sampler.sampler.indices.index(ind) for ind in sample_inds[phase][0:4]]
                fig = Trainer.show_imgs(inputs=inputs_phase, outputs=outputs_phase, predictions=predictions_phase,
                                        targets=targets_phase, inds=sample_inds_epoch)
                figname = 'image_examples_'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'))
                writer.add_figure(figname + phase, fig, epoch)

                fig = Trainer.show_classification_matrix(targets=targets_phase, predictions=predictions_phase, metrics=metrics)
                figname = 'targets_outputs_correct_'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'))
                writer.add_figure(figname + phase, fig, epoch)

                if self.save & (phase == 'train'):
                    print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Writing model graph ... ')
                    writer.add_graph(self.model, inputs)

                    print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Saving model state... ')
                    self.model.epoch = torch.nn.Parameter(torch.tensor(epoch), requires_grad=False)
                    self.model.iteration = torch.nn.Parameter(torch.tensor(it), requires_grad=False)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, os.path.join(self.log_root, epoch_root, 'model_state_dict'))
                    torch.save({
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.log_root, 'optimizer_state_dict'))

                if phase == 'test':
                    writer.add_pr_curve(
                        'pr_curve_test', labels=targets_phase, predictions=np.exp(outputs_phase[:, 1]), global_step=it,
                        num_thresholds=50)



        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Finished training ... ')

        writer.close()
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Closed writer ... ')

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
    def show_imgs(inputs, outputs, predictions, targets, inds, class_idx=1):
        fig, axs = plt.subplots(2, len(inds), figsize=(3*len(inds), 6))
        for i, idx in enumerate(inds):
            input = inputs[i, 0, :, :].squeeze()
            axs[0, idx].imshow(input, cmap='gray')
            axs[0, idx].axis('off')
            output = np.tile(np.exp((outputs[i][class_idx])), (int(input.shape[0]/3), int(input.shape[1])))
            prediction = np.tile(int(predictions[i]), (int(input.shape[0]/3), int(input.shape[1])))
            target = np.tile(int(targets[i]), (int(input.shape[0]/3), int(input.shape[1])))
            fused = np.concatenate((output, prediction, target), axis=0)
            axs[1, idx].imshow(fused, cmap='gray', vmin=0, vmax=1)
            axs[1, idx].text(0.5, 0.8, 'output (class {:d}): {:01.2f}'.format(class_idx, np.exp((outputs[i][class_idx]))),
                             transform=axs[1, idx].transAxes, ha='center', va='center', c=[0.8, 0.8, 0.2])
            axs[1, idx].text(0.5, 0.5, 'prediction class: {:d}'.format(int(predictions[i])),
                             transform=axs[1, idx].transAxes, ha='center', va='center', c=[0.5, 0.5, 0.5])
            axs[1, idx].text(0.5, 0.2, 'target class: {:d}'.format(int(targets[i])),
                             transform=axs[1, idx].transAxes, ha='center', va='center', c=[0.5, 0.5, 0.5])
            axs[1, idx].axis('off')

        plt.tight_layout()

        return fig

    @staticmethod
    def compute_metrics(targets, predictions):

        metrics = dict()
        metrics['inds_true_pos'] = (predictions == 1) & (targets == 1)
        metrics['sum_true_pos'] = sum(metrics['inds_true_pos'])
        metrics['inds_false_pos'] = (predictions == 1) & (targets == 0)
        metrics['sum_false_pos'] = sum(metrics['inds_false_pos'])
        metrics['inds_true_neg'] = (predictions == 0) & (targets == 0)
        metrics['sum_true_neg'] = sum(metrics['inds_true_neg'])
        metrics['inds_false_neg'] = (predictions == 0) & (targets == 1)
        metrics['sum_false_neg'] = sum(metrics['inds_false_neg'])

        metrics['accuracy'] = (metrics['sum_true_pos'] + metrics['sum_true_neg']) / len(targets)

        if metrics['sum_true_pos'] + metrics['sum_true_neg'] > 0:
            metrics['true_pos_rate'] = metrics['sum_true_pos'] / (metrics['sum_true_pos'] + metrics['sum_true_neg'])
            metrics['true_neg_rate'] = metrics['sum_true_neg'] / (metrics['sum_true_pos'] + metrics['sum_true_neg'])
        else:
            metrics['true_pos_rate'] = np.NaN
            metrics['true_neg_rate'] = np.NaN

        if metrics['sum_true_pos'] + metrics['sum_true_neg'] > 0:
            metrics['false_pos_rate'] = metrics['sum_false_pos'] / (metrics['sum_false_pos'] + metrics['sum_false_neg'])
            metrics['false_neg_rate'] = metrics['sum_false_neg'] / (metrics['sum_false_pos'] + metrics['sum_false_neg'])
        else:
            metrics['false_pos_rate'] = np.NaN
            metrics['false_neg_rate'] = np.NaN

        if (metrics['sum_true_pos'] + metrics['sum_false_pos']) > 0:
            metrics['precision'] = metrics['sum_true_pos'] / (metrics['sum_true_pos'] + metrics['sum_false_pos'])
        else:
            metrics['precision'] = np.NaN

        if (metrics['sum_true_pos'] + metrics['sum_false_neg']) > 0:
            metrics['recall'] = metrics['sum_true_pos'] / (metrics['sum_true_pos'] + metrics['sum_false_neg'])
        else:
            metrics['recall'] = np.NaN

        return metrics

    @staticmethod
    def show_classification_matrix(targets, predictions, metrics):

        targets_pr = targets.copy().astype(int)
        predictions_pr = predictions.copy().astype(int)
        correct_pr = ((targets_pr == 0) == (predictions_pr == 0)) | ((targets_pr == 1) == (predictions_pr == 1)) + 2
        code_pr = targets_pr.copy()
        code_pr[metrics['inds_true_neg']] = 4
        code_pr[metrics['inds_true_pos']] = 5
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
        fig, axs = plt.subplots(figsize=(12, 6))
        axs.matshow(mat, cmap=cmap, vmin=0, vmax=7)
        axs.set_yticks([0, 1, 2, 3])
        axs.set_yticklabels(['targets', 'outputs', 'accuracy', 'confusion'])
        axs.set_aspect(10)
        bbox = axs.get_position().bounds
        axs2 = plt.axes((bbox[0], 0.1, bbox[2], 0.2), sharex=axs)
        axs2.text(0.00, 1.00, 'target|output', c=(0.2, 0.2, 0.2), weight='bold', transform=axs2.transAxes)
        axs2.text(0.017, 0.75, 'artifact', c=(0.2, 0.2, 0.2), backgroundcolor=colors[1], transform=axs2.transAxes)
        axs2.text(0.017, 0.50, 'no artifact', c=(0.8, 0.8, 0.8), backgroundcolor=colors[0], transform=axs2.transAxes)
        axs2.text(0.27, 1.00, 'accuracy', c=(0.2, 0.2, 0.2), weight='bold', transform=axs2.transAxes)
        axs2.text(0.20, 0.75, 'frac correct:   {:03d}/{:03d}={:01.2f}'.format(metrics['sum_true_pos'] +
                  metrics['sum_true_neg'], len(targets), (metrics['sum_true_pos'] +
                  metrics['sum_true_neg'])/len(targets)), c=(0.2, 0.2, 0.2), backgroundcolor=colors[3],
                  transform=axs2.transAxes)
        axs2.text(0.20, 0.50, 'frac incorrect: {:03d}/{:03d}={:01.2f}'.format(metrics['sum_false_pos'] +
                  metrics['sum_false_neg'], len(targets), (metrics['sum_false_pos'] +
                  metrics['sum_false_neg'])/len(targets)), c=(0.2, 0.2, 0.2), backgroundcolor=colors[2],
                  transform=axs2.transAxes)
        axs2.text(0.60, 1.00, 'confusion', c=(0.2, 0.2, 0.2), weight='bold', transform=axs2.transAxes)
        axs2.text(0.50, 0.75, 'TP: {:01.2f}'.format(metrics['true_pos_rate']), c=(0.8, 0.8, 0.8),
                  backgroundcolor=colors[5], transform=axs2.transAxes)
        axs2.text(0.50, 0.50, 'TN: {:01.2f}'.format(metrics['true_neg_rate']), c=(0.8, 0.8, 0.8),
                  backgroundcolor=colors[4], transform=axs2.transAxes)
        axs2.text(0.60, 0.75, 'FP: {:01.2f}'.format(metrics['false_pos_rate']), c=(0.2, 0.2, 0.2),
                  backgroundcolor=colors[6], transform=axs2.transAxes)
        axs2.text(0.60, 0.50, 'FN: {:01.2f}'.format(metrics['false_neg_rate']), c=(0.2, 0.2, 0.2),
                  backgroundcolor=colors[7], transform=axs2.transAxes)
        axs2.text(0.70, 0.75, 'Precision: {:01.2f}'.format(metrics['precision']), c=(0.2, 0.2, 0.2),
                  backgroundcolor=(0.7, 0.7, 0.7), transform=axs2.transAxes)
        axs2.text(0.70, 0.50, 'Recall:    {:01.2f}'.format(metrics['recall']), c=(0.2, 0.2, 0.2),
                  backgroundcolor=(0.7, 0.7, 0.7), transform=axs2.transAxes)
        axs2.axis('off')

        return fig




