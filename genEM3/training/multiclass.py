import os
import datetime
from typing import List, Union, Dict, Sequence
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.metrics as sk_metrics
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice
from torch.utils.data import Sampler, Subset, DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.functional import nll_loss

from genEM3.util import gpu
from genEM3.training.metrics import Metrics
from genEM3.data.wkwdata import WkwData


class Trainer:

    def __init__(self,
                 run_name: str,
                 run_root: str,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.MSELoss,
                 data_loaders: Union[List, Dict],
                 num_epoch: int = 100,
                 log_int: int = 10,
                 device: str = 'cpu',
                 save: bool = False,
                 save_int: int = 1,
                 resume: bool = False,
                 gpu_id: int = None,
                 class_target_value: List = None):
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
        # can only get the lengths when a single set of data loaders are used
        if isinstance(data_loaders, dict):
            self.data_lengths = dict(zip(self.data_loaders.keys(), [len(loader) for loader in self.data_loaders]))
        else:
            self.data_lengths = {}
        if save:
            if not os.path.exists(self.log_root):
                os.makedirs(self.log_root)
        # The information regarding the class indices only useful if discrete target
        self.class_target_value = class_target_value

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
        artefact_idx = self.class_target_value._fields.index('artefact')
        epoch = int(self.model.epoch) + 1
        batch_index = int(self.model.iteration)
        sample_count_df = pd.DataFrame(np.zeros([2, 2], dtype=np.int64), columns=self.class_target_value._fields, index=('No', 'Yes'))
        for epoch in range(epoch, epoch + self.num_epoch):
            if isinstance(self.data_loaders, list):
                # each element of the list is a data loader for an epoch
                loader_change_interval = self.num_epoch / len(self.data_loaders)
                division_index, _ = divmod(epoch, loader_change_interval)
                # Make sure that the index does not exceed the length of the data_loader list
                index = round(min(division_index, len(self.data_loaders)-1))
                epoch_data_loaders = self.data_loaders[index]
            else:
                # same dataloaders for all epochs
                epoch_data_loaders = self.data_loaders
            # Dictionary (of dictionaries) to collect four metrics from different phases for tensorboard
            epoch_metric_names = ['epoch_loss', 'epoch_accuracy', 'precision_for_debris', 'recall_for_debris']
            epoch_metric_dict = {metric_name: dict.fromkeys(epoch_data_loaders.keys()) for metric_name in epoch_metric_names}
            epoch_root = 'epoch_{:02d}'.format(epoch)
            if not os.path.exists(os.path.join(self.log_root, epoch_root)):
                os.makedirs(os.path.join(self.log_root, epoch_root))
            for phase in ['val']: # epoch_data_loaders.keys()
                cur_loader = epoch_data_loaders[phase]
                num_batches = len(cur_loader)
                total_sample_counter = 0
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                epoch_loss = 0.0
                running_loss = 0.0
                correct_sum = 0
                batch_idx_start = 0

                num_items_phase = len(cur_loader.batch_sampler.sampler)
                inputs_phase = -np.ones((num_items_phase, 1, 140, 140)).astype(float)
                outputs_phase = -np.ones((num_items_phase, self.model.classifier.num_output)).astype(float)
                predictions_phase = -np.ones((num_items_phase, self.model.classifier.num_output)).astype(int)
                targets_phase = -np.ones((num_items_phase, self.model.classifier.num_output)).astype(int)
                correct_phase = -np.ones((num_items_phase, self.model.classifier.num_output)).astype(int)

                sample_ind_phase = []
                for i, data in enumerate(cur_loader):
                    batch_index += 1
                    # copy input and targets to the device object
                    inputs = data['input'].to(self.device)
                    cur_batch_size = inputs.shape[0]
                    targets = data['target'].float().to(self.device)
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
                    
                    # Focus on the debris decision
                    predicted_classes = (torch.sigmoid(outputs.detach()).numpy() > 0.5).astype(int)
                    target_classes = targets.detach().numpy().astype(int)
                    correct_classes = (predicted_classes == target_classes).astype(int)
                    correct_sum += correct_classes.sum(axis=0)[artefact_idx]

                    if i > 0:
                        batch_idx_start = batch_idx_end
                    batch_idx_end = batch_idx_start + cur_batch_size
                    inputs_phase[batch_idx_start:batch_idx_end, :, :, :] = inputs.detach().numpy()
                    outputs_phase[batch_idx_start:batch_idx_end, :] = outputs.detach().numpy()
                    predictions_phase[batch_idx_start:batch_idx_end, :] = predicted_classes
                    targets_phase[batch_idx_start:batch_idx_end] = target_classes
                    correct_phase[batch_idx_start:batch_idx_end] = correct_classes

                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    # Gather number of each class in mini batch
                    total_sample_counter += cur_batch_size
                    non_zero_count = np.count_nonzero(target_classes, axis=0)
                    cur_sample_count = np.vstack((cur_batch_size-non_zero_count, non_zero_count))
                    assert (cur_sample_count.sum(axis=0) == cur_batch_size).all(), 'Sum to batch size check failed'
                    sample_count_df = sample_count_df + cur_sample_count
                    # logging for the mini-batches
                    if i % self.log_int == 0:
                        running_loss_log = float(running_loss) / batch_index
                        running_accuracy_log = float(correct_sum) / batch_idx_end
                        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')' + ' Phase: ' + phase +
                              ', epoch: {}, batch: {}, running loss: {:0.4f}, running accuracy: {:0.3f} '.
                              format(epoch, i, running_loss_log, running_accuracy_log))
                        writer.add_scalars('running_loss', {phase: running_loss_log}, batch_index)
                        writer.add_scalars('running_accuracy', {phase: running_accuracy_log}, batch_index)

                # Number of samples checked two ways
                assert total_sample_counter == num_items_phase
                # Fraction for each class of target
                class_fraction_df = sample_count_df / total_sample_counter
                assert np.isclose(class_fraction_df.sum(), 1.0).all(), 'All fraction sum to 1.0 failed'
                # the index for positive examples in each class
                with_index = 'Yes'
                fraction_positive_dict = class_fraction_df.loc[with_index].to_dict()
                writer.add_scalars(f'Fraction_with_target_{phase}', fraction_positive_dict, epoch)
                # calculate epoch loss and accuracy average over batch samples
                # Epoch error measures:
                epoch_loss_log = float(epoch_loss) / num_batches
                epoch_accuracy_log = float(correct_sum) / num_items_phase
                print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')' + ' Phase: ' + phase +
                      ', epoch: {}: epoch loss: {:0.4f}, epoch accuracy: {:0.3f} '.
                      format(epoch, epoch_loss_log, epoch_accuracy_log))
                error_measures = sk_metrics.precision_recall_fscore_support(targets_phase[:,artefact_idx], predictions_phase[:,artefact_idx], zero_division=0)
                debris_idx = 1
                cur_metrics = [epoch_loss_log, epoch_accuracy_log, error_measures[0][debris_idx], error_measures[1][debris_idx]]
                for i, metric_name in enumerate(epoch_metric_names):
                    epoch_metric_dict[metric_name][phase] = cur_metrics[i]

                # Confusion matrix
                confusion_matrix_phase = sk_metrics.confusion_matrix(targets_phase[:,artefact_idx], predictions_phase[:,artefact_idx])
                fig_confusion = self.plot_confusion_matrix(confusion_matrix_phase)
                figname_confusion = 'confusion_matrix_'
                fig_confusion.savefig(os.path.join(self.log_root, epoch_root, figname_confusion + phase + '.png'), dpi=300)
                writer.add_figure(figname_confusion + phase, fig_confusion, epoch)
                
                # confusion normalized
                fig_confusion_norm = self.plot_confusion_matrix(confusion_matrix_phase, normalize_dim=1)
                figname_confusion_norm = 'confusion_matrix_normalized_'
                fig_confusion_norm.savefig(os.path.join(self.log_root, epoch_root, figname_confusion_norm + phase + '.png'), dpi=300)
                writer.add_figure(figname_confusion_norm + phase, fig_confusion_norm, epoch)

                # Example images
                fig = Trainer.show_imgs(inputs=inputs_phase, 
                                        outputs=outputs_phase,
                                        predictions=predictions_phase,
                                        targets=targets_phase,
                                        target_names=self.get_class_names(),
                                        sample_ind=sample_ind_phase,
                                        class_idx=debris_idx,
                                        )
                figname = 'Image_examples_with_highest_loss_'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname + '_' + phase + '.png'), dpi=300)
                writer.add_figure(figname + phase, fig, epoch)

                # Precision/Recall curves
                for c in self.class_target_value:
                    cur_binary_targets = (targets_phase==c[0]).astype(int)
                    writer.add_pr_curve(
                        'pr_curve_'+phase+'_'+c[1], labels=cur_binary_targets, predictions=np.exp(outputs_phase[:, c[0]]), global_step=epoch,
                        num_thresholds=50)
                # save model
                if self.save & (phase == 'train') & (epoch % self.save_int == 0):
                    print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Writing model graph ... ')
                    # writer.add_graph(self.model, inputs)

                    print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Saving model state... ')
                    self.model.epoch = torch.nn.Parameter(torch.tensor(epoch), requires_grad=False)
                    self.model.iteration = torch.nn.Parameter(torch.tensor(batch_index), requires_grad=False)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, os.path.join(self.log_root, epoch_root, 'model_state_dict'))
                    torch.save({
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.log_root, 'optimizer_state_dict'))
            # write the epoch related metrics to the tensorboard
            for metric_name in epoch_metric_names:
                writer.add_scalars(metric_name, epoch_metric_dict[metric_name], epoch)
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Finished training ... ')

        writer.close()
        print('(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ') Closed writer ... ')

    def get_class_index(self, name: str='Debris'):
        """Get the index from the class information list"""
        target_index = [c[0] for c in self.class_target_value if c[1]==name]
        assert len(target_index) == 1
        return target_index[0]

    
    def get_class_names(self):
        """Get the index from the class information list"""
        target_names = [c[1] for c in self.class_target_value]
        return target_names

    def plot_confusion_matrix(self, confusion_matrix, normalize_dim: int = None):
        """Plot the confusion matrix"""
        # Get the group names
        group_names = [c[1] for c in self.class_target_value]
        fig, ax = plt.subplots(figsize=(5,5))
        # Normalize data if requested. The values are rounded to 2 decimal points
        if normalize_dim is not None:
            sums_along_dim = confusion_matrix.sum(axis=normalize_dim, keepdims=True)
            confusion_matrix = np.around(confusion_matrix / sums_along_dim,2)
            im = ax.imshow(confusion_matrix, cmap='gray',vmin=0,vmax=1)
        else:
            im = ax.imshow(confusion_matrix, cmap='gray')
        
        # We want to show all ticks...
        num_groups = len(group_names)
        ax.set_xticks(np.arange(num_groups))
        ax.set_yticks(np.arange(num_groups))
        # ... and label them with the respective list entries
        ax.set_xticklabels(group_names)
        ax.set_yticklabels(group_names)
        # Give axis labels as well:
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('Human label class')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        max_val = confusion_matrix.max() 
        # Loop over data dimensions and create text annotations.
        for i in range(num_groups):
            for j in range(num_groups):
                entry = confusion_matrix[i, j]
                if entry > float(max_val)/2:
                    color = 'k'
                else:
                    color = 'w'
                text = ax.text(j, i, entry,
                               ha="center", va="center", color=color)

        ax.set_title("Confusion matrix")
        fig.colorbar(im)
        fig.tight_layout()
        return fig

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
    def show_imgs(inputs, outputs, predictions, targets, sample_ind, target_names, class_idx=1, plot_ind=None):
        # Get the top 5 losses if the indices are not give
        if plot_ind is None:
            losses = nll_loss(torch.from_numpy(outputs) , torch.from_numpy(targets) , reduction='none')
            (values_sorted, indices_sorted) = losses.sort(descending=True)
            plot_ind = indices_sorted[range(5)].numpy()

        fig, axs = plt.subplots(2, len(plot_ind), figsize=(3 * len(plot_ind), 6))
        for i, sample_idx in enumerate(np.asarray(sample_ind)[plot_ind]):
            input = inputs[i, 0, :, :].squeeze()
            target_shape = (int(input.shape[0]/3), int(input.shape[1]))
            axs[0, i].imshow(input, cmap='gray')
            axs[0, i].axis('off')

            # Create images for the target values
            output = np.tile(np.exp((outputs[i][class_idx])), target_shape)
            # output values above 1 are shown as zeros (myelin is added to the clean group)
            if predictions[i] > 1:
                gray_val_pred = 0
            else:
                gray_val_pred = int(predictions[i])
            if targets[i] > 1:
                gray_val_target = 0
            else:
                gray_val_target = int(targets[i])
            prediction = np.tile(gray_val_pred, target_shape)
            target = np.tile(gray_val_target, target_shape)
            fused = np.concatenate((output, prediction, target), axis=0)
            axs[1, i].imshow(fused, cmap='gray_r', vmin=0, vmax=1)
            axs[1, i].text(0.5, 1.0, 'Probability, Non-Myl/deb/Myl:\n{}'.format(np.exp(outputs[i]).round(2).tolist()),
                           transform=axs[1, i].transAxes, ha='center', va='center', c=[0.8, 0.8, 0.2])
            axs[1, i].text(0.5, 0.80, 'sample_idx: {}'.format(sample_idx),
                           transform=axs[1, i].transAxes, ha='center', va='center', c=[0.8, 0.8, 0.2])
            axs[1, i].text(0.5, 0.70, 'Probability {}: {:01.2f}'.format(target_names[class_idx], np.exp((outputs[i][class_idx]))),
                             transform=axs[1, i].transAxes, ha='center', va='center', c=[0.8, 0.8, 0.2])
            axs[1, i].text(0.5, 0.5, 'Prediction class: {}'.format(target_names[int(predictions[i])]),
                             transform=axs[1, i].transAxes, ha='center', va='center', c=[0.5, 0.5, 0.5])
            axs[1, i].text(0.5, 0.2, 'Target class: {}'.format(target_names[int(targets[i])]),
                             transform=axs[1, i].transAxes, ha='center', va='center', c=[0.5, 0.5, 0.5])
            axs[1, i].axis('off')

        plt.tight_layout()

        return fig


class subsetWeightedSampler(Sampler):
    """Sampler for generating a weighted random subset of the given dataset  

    Arguments:
        wkw_dataset (WkwData): dataset to sample (subclass of torch Dataset)
        subset_indices: List of sample indices used to create the sampler
        imbalance_factor: ratio of clean/debris classes in generated sampler
        verbose: whether or not to print debug info
    """

    def __init__(self,
                 wkw_dataset: WkwData,
                 subset_indices: List[np.int64],
                 fraction_debris: float,
                 artefact_dim: int,
                 verbose: bool = False):
        self.wkw_dataset = wkw_dataset
        self.subset_indices = subset_indices
        self.frac_clean_debris = np.asarray([1 - fraction_debris, fraction_debris])
        self.artefact_dim = artefact_dim
        # Get the target (debris vs. clean) for each sample
        total_sample_range = iter(subset_indices)
        self.index_set = set(subset_indices)
        # check uniqueness of indices
        assert len(self.index_set) == len(self.subset_indices)
        self.target_class = np.asarray([wkw_dataset.get_target_from_sample_idx(sample_idx) for sample_idx in total_sample_range], dtype=np.int64)
        self.artefact_targets = self.target_class[:, artefact_dim]
        if verbose:
            self.report_original_numbers()     

        # Use the inverse of the number of samples as weight to create balance
        self.class_sample_count = np.array(
            [len(np.where(self.artefact_targets == t)[0]) for t in np.unique(self.artefact_targets)])

        # Subset dataset
        self.sub_dataset = Subset(wkw_dataset, subset_indices)

    def __iter__(self):
        """Method called when iter() calls the sampler. Returns a iterator over the samples.
        This method directly returns the iterator from weightedRandomSampler of pytorch.
        """
        weight = self.frac_clean_debris / self.class_sample_count
        samples_weight = np.array([weight[t[self.artefact_dim]] for t in self.target_class])
        # Create the weighted sampler
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return iter(sampler)

    def __len__(self):
        """the length of the sampler (length of the sub dataset)"""
        return len(self.sub_dataset)

    def report_original_numbers(self):
        """print the sample imbalance"""
        print('Target balance for original train set clean/debris: {}/{}'.format(
            len(np.where(self.artefact_targets == 0)[0]), len(np.where(self.artefact_targets == 1)[0])))

    @classmethod
    def get_data_loaders(cls,
                         dataset: WkwData,
                         fraction_debris: float,
                         artefact_dim: int,
                         test_dataset: WkwData = None,
                         batch_size: int = 256,
                         num_workers: int = 0) -> Dict[str, torch.utils.data.DataLoader]:
        """Generate the pytorch data loaders for the training and validation subsets of the dataset. 
        The balance of two target classes are controled by imbalance factor.
        Arguments:
            dataset: Pytorch dataset with train and validation indices
            fraction_debris: The fraction of debris in the dataset
            batch_size and num_workers are inputs to data loader
        Output:
            data_loaders: dictionary of the pytroch data loaders
        """
        index_names = {
           "train": "data_train_inds",
           "val": "data_validation_inds",
           "test": "data_test_inds"}
        data_loaders = dict.fromkeys(index_names)
        for key in index_names:
            cur_indices = getattr(dataset, index_names.get(key))
            # Only get the data loader if the indices is not empty. Otherwise leave the entry as None in data_loaders
            if bool(cur_indices):
                # Only for training data create a balance controlled dataset
                # The test and validation are random permutation of the same set of data without any class balancing
                if key == 'train':
                    cur_sampler = cls(dataset, cur_indices, fraction_debris=fraction_debris, artefact_dim=artefact_dim)
                    data_loaders[key] = torch.utils.data.DataLoader(
                        dataset=cur_sampler.sub_dataset, batch_size=batch_size, num_workers=num_workers, sampler=cur_sampler,
                        collate_fn=dataset.collate_fn)
                elif key in ['val', 'test']:
                    data_loaders[key] = torch.utils.data.DataLoader(
                        dataset=dataset, batch_size=batch_size, num_workers=num_workers, 
                        sampler=SubsetRandomSampler(cur_indices),
                        collate_fn=dataset.collate_fn)
                else:
                    raise Exception(f'key not defined: {key}')
        # Use a sequential sampler for the 3 test boxes, test sampler/loader
        if test_dataset is not None:
            test_sampler = RandomSampler(test_dataset)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler,
                collate_fn=dataset.collate_fn)
            # Ignore the test dataset from above and replace it with the given test dataset. This was used when a separate dataset is used for testing
            data_loaders["test"] = test_loader
        return data_loaders

    @staticmethod
    def report_loader_composition(dataloader_dict: Dict,
                                  artefact_dim: int = 0,
                                  report_batch_data: bool = False):
        """
        Given a dictionary of data loaders, report on the fraction of clean/debris and 
        the number of repeating indices in each batch
        INPUT:
            dataloader_dict: dictionary of data loaders. Usually: train, val and test key-dataloader pairs
        """
        for key in dataloader_dict:
            total_debris = float(0)
            total_sample = float(0)
            fraction_debris = []
            fraction_repeat = []
            cur_dataloader = dataloader_dict[key]
            for i, data in enumerate(cur_dataloader):
                cur_batch_size = data['input'].shape[0]
                total_sample += cur_batch_size
                # check that all sample indices are part of the total indices
                batch_idx = data['sample_idx']
                assert len(batch_idx) == cur_batch_size, 'The index length does not match the input first dimension'
                batch_idx_set = set(batch_idx)
                repetition_num = cur_batch_size-len(batch_idx_set)
                cur_repetition_Frac = float(repetition_num) / float(cur_batch_size)
                fraction_repeat.append(cur_repetition_Frac * 100)
                y = data['target']
                clean_num = float((y[:, artefact_dim] == 0).sum())
                debris_num = float((y[:, artefact_dim] == 1).sum())
                total_debris += debris_num
                assert clean_num + debris_num == cur_batch_size, 'Sum check failed'
                cur_frac_debris = debris_num / cur_batch_size
                fraction_debris.append(cur_frac_debris * 100)
                if report_batch_data:
                    print(f'#####\nBatch of {key} Nr: {i+1}/{len(cur_dataloader)}, Batch size: {cur_batch_size}')
                    print(f'Batch fraction debris: {cur_frac_debris*100:.2f} %, Batch Fraction repeat: {cur_repetition_Frac*100:.2f} %\n#####')
            avg_fraction_debris = np.asarray(fraction_debris).mean()
            average_Frac_repetition = np.asarray(fraction_repeat).mean()
            # Grand average by summing all debris and sample numbers and dividing them
            total_frac_debris = total_debris / total_sample
            print(f'____\nDataLoader type: {key}, Num batches: {len(cur_dataloader)}, Batch size: {cur_dataloader.batch_size}')
            print(f'Average Fraction debris: {avg_fraction_debris:.2f}%\nAverage Repetition Fraction:{average_Frac_repetition:.2f}%')
            print(f'Total debris fraction: {total_frac_debris * 100:.2f} %\n_____')