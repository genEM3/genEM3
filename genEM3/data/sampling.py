from typing import List, Union, Dict 

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, Subset, RandomSampler, SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler

from genEM3.data.wkwdata import WkwData


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