import os
import datetime
from typing import List, Union, Dict 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import device as torchDevice
from torch.utils.data import Sampler, Subset, RandomSampler, SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler

from genEM3.util import gpu, io
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
                 resume_epoch: int = None,
                 gpu_id: int = None,
                 target_names: List = None):
        self.run_name = run_name
        self.run_root = run_root
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.log_int = log_int
        self.save = save
        self.save_int = save_int
        if resume_epoch is not None:
            self.resume = True
            self.resume_epoch = resume_epoch
        else:
            self.resume = False
            self.resume_epoch = resume_epoch
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
        self.target_names = target_names

    def train(self):
        # Load saved model if resume option selected
        if self.resume:
            print(Trainer.time_str() + ' Resuming training ... ')
            checkpoint = torch.load(os.path.join(self.log_root, self.get_epoch_root(self.resume_epoch), 'torch_model_optim.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(Trainer.time_str() + ' Starting training ... ')

        writer = SummaryWriter(self.log_root)
        self.model = self.model.to(self.device)
        epoch = int(self.model.epoch) + 1
        batch_counter = int(self.model.iteration)
        # Epoch loop
        for epoch in range(epoch, epoch + self.num_epoch):
            # Create logging directory
            epoch_root = self.get_epoch_root(epoch) 
            if not os.path.exists(os.path.join(self.log_root, epoch_root)):
                os.makedirs(os.path.join(self.log_root, epoch_root))
            # Select data loaders for the current epoch
            cur_epoch_loaders = self.get_epoch_loaders(epoch)
            # Dictionary (of dictionaries) to collect four metrics from different phases for tensorboard
            epoch_metric_names = ['epoch_loss', 'epoch_accuracy', 'epoch_precision', 'epoch_recall']
            epoch_metric_dict = {metric_name: dict.fromkeys(cur_epoch_loaders.keys()) for metric_name in epoch_metric_names}
            # Loop over phases within one epoch [train, validation, test]
            for phase in cur_epoch_loaders.keys():
                # Select training state of the NN model
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                # Select Loader
                cur_loader = cur_epoch_loaders[phase]
                # Number of samples
                columns = self.target_names.columns
                sample_count_df = pd.DataFrame(np.zeros([2, len(columns)],
                                               dtype=np.int64),
                                               columns=columns,
                                               index=('No', 'Yes'))
                num_samples = len(cur_loader.batch_sampler.sampler)
                total_sample_counter = 0
                num_target_class = self.model.classifier.num_output
                # initializing variables for keeping track of results for tensorboard reporting
                results_phase = self.init_results_phase(num_samples=num_samples, num_target_class=num_target_class)
                for i, data in enumerate(cur_loader):
                    batch_counter += 1
                    # Copy input and targets to the device object
                    inputs = data['input'].to(self.device)
                    type_indices = self.get_target_type_index()
                    targets = data['target'][:, type_indices].float().squeeze().to(self.device)
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets)
                    # Backward + Optimize(in training)
                    if phase == 'train':
                        loss.mean().backward()
                        self.optimizer.step()
                    
                    # Record results of the operation for the reporting
                    results_batch = self.get_results_batch(results_phase.keys(), data, loss, outputs)
                    # Aggregate results into a phase array for complete epoch reporting
                    cur_batch_size = inputs.shape[0]
                    nominal_batch_size = cur_loader.batch_size
                    results_phase, batch_idx_range = self.update_results_phase(results_batch=results_batch, 
                                                                               results_phase=results_phase,
                                                                               nominal_batch_size=nominal_batch_size,
                                                                               cur_batch_size=cur_batch_size,
                                                                               batch_idx=i)
                    # Gather number of each class in mini batch
                    total_sample_counter += cur_batch_size
                    non_zero_count = np.count_nonzero(results_batch['target'], axis=0)
                    cur_sample_count = np.vstack((cur_batch_size-non_zero_count, non_zero_count))
                    assert (cur_sample_count.sum(axis=0) == cur_batch_size).all(), 'Sum to batch size check failed'
                    sample_count_df = sample_count_df + cur_sample_count
                    # logging for the running loss and accuracy for each target class
                    if i % self.log_int == 0:
                        running_loss_log = results_phase['loss'][:batch_idx_range[1]].mean(axis=0)
                        running_accuracy = results_phase['correct'][:batch_idx_range[1]].mean(axis=0)
                        accuracy_dict = self.add_target_names(running_accuracy.round(3))
                        running_loss_dict = self.add_target_names(running_loss_log.round(3))
                        print(Trainer.time_str() + ' Phase: ' + phase +
                              f', epoch: {epoch}, batch: {i}, running loss: {running_loss_dict}, running accuracy: {accuracy_dict}')
                        writer.add_scalars(f'running_loss/{phase}', running_loss_dict, batch_counter)
                        writer.add_scalars(f'running_accuracy/{phase}', accuracy_dict, batch_counter)

                # Number of samples in epoch checked two ways
                assert total_sample_counter == num_samples
                # Make sure no -1s left in the phase results (excluding input which throws errors)
                for key in ['loss', 'output_prob', 'prediction', 'target', 'correct']:
                    assert not (results_phase[key] == -1).any()
                # Fraction for each class of target
                class_fraction_df = sample_count_df / num_samples
                assert np.isclose(class_fraction_df.sum(), 1.0).all(), 'All fraction sum to 1.0 failed'
                # the index for positive examples in each class
                with_index = 'Yes'
                fraction_positive_dict = class_fraction_df.loc[with_index].to_dict()
                writer.add_scalars(f'Fraction_with_target/{phase}', fraction_positive_dict, epoch)
                # calculate epoch loss and accuracy average over batch samples
                # Epoch error measures
                epoch_loss_log = results_phase['loss'].mean(axis=0)
                epoch_loss_dict = self.add_target_names(epoch_loss_log.round(3))
                epoch_accuracy_log = results_phase['correct'].mean(axis=0)
                epoch_acc_dict = self.add_target_names(epoch_accuracy_log.round(3))
                print(Trainer.time_str() + ' Phase: ' + phase +
                      f', epoch: {epoch}: epoch loss: {epoch_loss_dict}, epoch accuracy: {epoch_acc_dict}')
                
                # Pickle important results dict elements: loss, output_prob and dataset_indices
                dict_to_save = {key: results_phase[key] for key in ['loss', 'output_prob', 'dataset_indices']}
                io.save_dict(dict_to_save, os.path.join(self.log_root, epoch_root, 'results_saved.pkl'))

                # Precision, recall, accuracy and loss 
                precision, recall, _, num_pos = sk_metrics.precision_recall_fscore_support(results_phase['target'].squeeze(),
                                                                                           results_phase['prediction'].squeeze(),
                                                                                           zero_division=0)
               
                # The metrics function returns the result for both positive and negative labels when operated with
                # a single target type. When the task is a multilabel decision it only returns the positive label results
                if num_target_class == 1:
                    precision = [precision[1]]
                    recall = [recall[1]]
                    num_pos = num_pos[1]
                assert (np.asarray(sample_count_df.loc['Yes']) == num_pos).all(), 'Number of positive samples matching failed'
                cur_metrics = [epoch_loss_dict, epoch_acc_dict,
                               self.add_target_names(precision), self.add_target_names(recall)]
                for i, metric_name in enumerate(epoch_metric_names):
                    epoch_metric_dict[metric_name][phase] = cur_metrics[i]
                
                # Confusion matrix Figure
                if num_target_class == 1:
                    confusion_matrix = sk_metrics.confusion_matrix(results_phase['target'].squeeze(),
                                                                   results_phase['prediction'].squeeze())
                elif num_target_class > 1:
                    confusion_matrix = sk_metrics.multilabel_confusion_matrix(results_phase['target'],
                                                                              results_phase['prediction'])
                else:
                    raise Exception('number of target classes is negative')
                fig_confusion_norm = self.plot_confusion_matrix(confusion_matrix)
                figname_confusion = 'Confusion_matrix'
                fig_confusion_norm.savefig(os.path.join(self.log_root,
                                                        epoch_root,
                                                        figname_confusion + phase + '.png'),
                                           dpi=300)
                writer.add_figure(f'{figname_confusion}/{phase}', fig_confusion_norm, epoch)

                # Images with highest loss in each target type (Myelin and artefact currently)
                fig = self.show_imgs(results_phase=results_phase)
                figname_examples = 'Examples_with_highest_loss'
                fig.savefig(os.path.join(self.log_root, epoch_root, figname_examples + '_' + phase + '.png'), dpi=300)
                writer.add_figure(f'{figname_examples}/{phase}', fig, epoch)

                # Precision/Recall curves
                for i, t_type in enumerate(self.target_names):
                    writer.add_pr_curve(f'{t_type}/{phase}',
                                        labels=results_phase.get('target')[:, i],
                                        predictions=results_phase.get('output_prob')[:, i],
                                        global_step=epoch,
                                        num_thresholds=100)

                # save model
                if self.save & (phase == 'train') & (epoch % self.save_int == 0):
                    print(Trainer.time_str() + ' Writing model graph ... ')
                    # writer.add_graph(self.model, inputs)
                    print(Trainer.time_str() + ' Saving model state... ')
                    self.model.epoch = torch.nn.Parameter(torch.tensor(epoch), requires_grad=False)
                    self.model.iteration = torch.nn.Parameter(torch.tensor(batch_counter), requires_grad=False)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.log_root, epoch_root, 'torch_model_optim.pth'))

            # write the epoch related metrics to the tensorboard
            for metric_name in epoch_metric_names:
                cur_metric = epoch_metric_dict[metric_name]
                for ph in cur_metric:
                    cur_metric_phase = {f'{ph}_{t_type}': val for t_type, val in cur_metric[ph].items()}
                    writer.add_scalars(metric_name, cur_metric_phase, epoch)
        print(Trainer.time_str() + ' Finished training ... ')
        writer.close()
        print(Trainer.time_str() + ' Closed writer ... ')

    def get_epoch_loaders(self, epoch):
        """
        Return the data loader. From a list (of dicts) or an individual dict
        """
        if isinstance(self.data_loaders, list):
            # Case: List of data loaders. Each loader is a dictionary with fields as phases.
            # Find which element of the list to choose based on change interval
            loader_change_interval = self.num_epoch / len(self.data_loaders)
            division_index, _ = divmod(epoch, loader_change_interval)
            # Make sure that the index does not exceed the length of the data_loader list
            index = round(min(division_index, len(self.data_loaders)-1))
            epoch_loaders = self.data_loaders[index]
        elif isinstance(self.data_loaders, dict):
            # Case: a single dictionary of loaders for epochs
            epoch_loaders = self.data_loaders
        else:
            raise Exception(f'Loader type not defined: {type(self.data_loaders)}')

        return epoch_loaders

    def init_results_phase(self, num_samples: int, num_target_class: int):
        """
        initialize the dictionary for gathering the results in each epoch
        """ 
        patch_size = self.data_loaders[0]['val'].dataset.input_shape[:2]
        results_phase = {'input': -np.ones((num_samples, 1, *patch_size)).astype(float),
                         'output_prob': -np.ones((num_samples, num_target_class)).astype(float),
                         'loss': -np.ones((num_samples, num_target_class), dtype=np.float32),
                         'prediction': -np.ones((num_samples, num_target_class)).astype(int),
                         'target': -np.ones((num_samples, num_target_class)).astype(int),
                         'correct': -np.ones((num_samples, num_target_class)).astype(int),
                         'dataset_indices': -np.ones(num_samples).astype(int)}
        return results_phase

    def get_results_batch(self, dict_keys, data, loss, outputs):
        """
        Create a dictionary that holds the results for the current batch
        """
        type_indices = self.get_target_type_index()
        results_batch = dict.fromkeys(dict_keys)
        results_batch['loss'] = Trainer.convert2numpy(loss)
        results_batch['input'] = Trainer.convert2numpy(data['input'])
        results_batch['output_prob'] = Trainer.convert2numpy(Trainer.sigmoid(outputs))
        results_batch['target'] = Trainer.convert2numpy(data['target'][:, type_indices].squeeze())
        decision_thresh = 0.5
        results_batch['prediction'] = (results_batch['output_prob'] > decision_thresh).astype(int)
        results_batch['correct'] = (results_batch['prediction'] == results_batch['target']).astype(int)
        results_batch['dataset_indices'] = np.asarray(data['sample_idx'])

    @staticmethod
    def update_results_phase(results_phase: dict,
                             results_batch: dict,
                             nominal_batch_size: int,
                             cur_batch_size: int,
                             batch_idx: int):
        """
        Update the entries in results for the whole epoch using the batch results
        """
        batch_idx_range = [batch_idx * nominal_batch_size, (batch_idx * nominal_batch_size) + cur_batch_size]
        for key in results_phase:
            # Add additional dimension if the array is 1d
            if results_batch[key].ndim == 1:
                results_batch[key] = np.expand_dims(results_batch[key], axis=1)
            results_phase[key][batch_idx_range[0]:batch_idx_range[1]] = results_batch[key]
        return results_phase, batch_idx_range

    def add_target_names(self, array):
        """
        Creates a dictionary with each element of array corresponding to the target names
        """
        assert len(self.target_names.columns) == len(array)
        return {key: val for key, val in zip(self.target_names.columns, array)}

    def get_target_type_index(self):
        """
        Get the indices of the target type based on the give target name dataframe
        """
        indices = []
        if 'artefact' in self.target_names.columns:
            indices.append(0)
        if 'myelin' in self.target_names.columns:
            indices.append(1)
        return indices

    def plot_confusion_matrix(self, confusion_matrix, normalize_dim: int = None):
        """Plot the confusion matrix"""
        # Get the group names
        matrix_shape = confusion_matrix.shape
        # Get the number of confusion matrices from the third dimension
        if len(matrix_shape) == 3:
            nrows = matrix_shape[0]
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(5, 5))
        elif len(matrix_shape) == 2:
            nrows = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(5, 5))
            # make axes indexable
            axes = [axes]
            # Expand the first dimension so that the indexing works the same as 3d array of confusion matrices
            confusion_matrix = np.expand_dims(confusion_matrix, axis=0)
        else:
            raise Exception('Matrix shape invalid')

        num_samples_per_conf = confusion_matrix.sum(axis=tuple(range(1, confusion_matrix.ndim)))
        assert max(num_samples_per_conf) == min(num_samples_per_conf), 'Sample numbers different for different target types'
        data_range = [0, num_samples_per_conf[0]]
        
        # Normalize data if requested. The values are rounded to 2 decimal points
        if normalize_dim is not None:
            sums_along_dim = confusion_matrix.sum(axis=normalize_dim, keepdims=True)
            confusion_matrix = np.divide(confusion_matrix,
                                         sums_along_dim,
                                         out=np.zeros_like(confusion_matrix, dtype=float),
                                         where=sums_along_dim != 0).round(3)
            data_range = [0, 1]
        # Loop over the targets and plot the confusion matrix
        for i in range(nrows):
            ax = axes[i]
            cur_conf_matrix = confusion_matrix[i]
            cur_im = ax.imshow(cur_conf_matrix, cmap='gray_r', vmin=data_range[0], vmax=data_range[1])
            ax.set_title(f'Confusion matrix: {self.target_names.columns[i]}')
            cur_target_names = list(self.target_names.iloc[:, i])
            # We want to show all ticks...
            num_groups = len(cur_target_names)
            ax.set_xticks(np.arange(num_groups))
            ax.set_yticks(np.arange(num_groups))
            # ... and label them with the respective list entries
            ax.set_xticklabels(cur_target_names)
            ax.set_yticklabels(cur_target_names)
            # Give axis labels as well:
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('Human label class')
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            max_val = data_range[1]
            # Loop over data dimensions and create text annotations.
            for i in range(num_groups):
                for j in range(num_groups):
                    entry = cur_conf_matrix[i, j]
                    if entry > float(max_val)/2:
                        color = 'w'
                    else:
                        color = 'k'
                    _ = ax.text(j, i, entry,
                                ha="center", va="center", color=color)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        if nrows == 2:
            cax = plt.axes([0.75, 0.1, 0.075, 0.8])
            plt.colorbar(cur_im, cax=cax)
            fig.tight_layout()
        elif nrows == 1:
            # single image
            plt.colorbar(cur_im, fraction=0.046, pad=0.04) 
        else:
            raise Exception('nrows one or two')

        return fig

    def show_imgs(self, results_phase: dict):
        """
        Show example images with highest loss
        """
        # convert to numpy array
        batch_sample_indexes = results_phase['dataset_indices'] 
        losses = results_phase['loss']
        sorted_ind = np.argsort(losses, axis=0)
        num_examples = 10
        num_targets = losses.shape[1]
        # loop over target types
        fig, axs = plt.subplots(nrows=num_targets, ncols=num_examples, figsize=(3 * num_examples, 3 * num_targets))
        # Make sure axs is two dimensional
        if axs.ndim == 1:
            axs = np.expand_dims(axs, axis=0)
        for i_target, t_name in enumerate(self.target_names):
            plot_ind = sorted_ind[-num_examples:, i_target]
            # Loop over [top 5] loss samples
            for idx, array_idx in enumerate(plot_ind):
                # Get the error metrics
                metric_keys = ['output_prob', 'target', 'loss']
                metric_val = dict.fromkeys(metric_keys)
                for field_name in metric_keys:
                    metric_val[field_name] = results_phase.get(field_name)[array_idx, i_target].round(2)
                # Plot
                cur_ax = axs[i_target, idx]
                input_im = results_phase.get('input')[array_idx, 0]
                cur_ax.imshow(input_im, cmap='gray')
                cur_ax.set_title(f'Target: {t_name}, Sample index: {batch_sample_indexes[array_idx]}\n{metric_val}', fontdict={'fontsize': 7})
                cur_ax.axis('off')
        plt.subplots_adjust(wspace=0.3)
        return fig

    @staticmethod
    def convert2numpy(tensor):
        """
        Convert tensor to a numpy array for tensorboard reporting
        """
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.numpy()
        return tensor

    @staticmethod
    def sigmoid(tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        return torch.sigmoid(tensor)

    @staticmethod
    def time_str():
        """
        Returns the time string for the reporting
        """
        return '(' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ')'

    @staticmethod
    def n1hw_to_n3hw(data):
        return data.cpu().repeat(1, 3, 1, 1)

    @staticmethod
    def show_img(inputs, outputs, idx):
        inputs = Trainer.convert2numpy(inputs)
        outputs = Trainer.convert2numpy(outputs)
        fig, axs = plt.subplots(1, 2, figsize=(4, 3))
        axs[0].imshow(inputs[idx].squeeze(), cmap='gray')
        axs[1].imshow(outputs[idx].squeeze(), cmap='gray')
        return fig

    @staticmethod
    def get_epoch_root(epoch):
        # returns the name of the epoch directory
        return 'epoch_{:02d}'.format(epoch)


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
