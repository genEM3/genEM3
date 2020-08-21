import os
import numpy as np
import pandas as pd
from typing import Union, Callable, Sequence
import torch

class Metrics:

    def __init__(self,
                 targets,
                 outputs,
                 output_prob_fn: Callable = None,
                 decision_th: float = 0.5,
                 sample_ind=None):

        if type(targets) == torch.Tensor:
            targets = targets.detach().numpy()
    
        if type(outputs) == torch.Tensor:
            outputs = outputs.detach().numpy()

        if output_prob_fn is None:
            output_prob_fn = lambda x: np.exp(x[:, 1])
            
        self.targets = targets
        self.outputs = outputs
        self.output_prob = output_prob_fn(outputs)
        self.output_prob_fn = output_prob_fn
        self.predictions = (self.output_prob > decision_th)
        self.predictions_correct = (self.predictions == self.targets)
        self.sample_ind = sample_ind
        
        self.metrics = self.compute(decision_th)

    def compute(self, decision_th: float = 0.5):

        metrics = dict()
        metrics['predictions'] = (self.output_prob > decision_th)
        metrics['predictions_correct'] = (metrics['predictions'] == self.targets)
        metrics['P'] = sum(self.targets == 1)
        metrics['N'] = sum(self.targets == 0)
        metrics['TP_idx'] = (metrics['predictions'] == 1) & (self.targets == 1)
        metrics['TP'] = sum(metrics['TP_idx'])
        metrics['FP_idx'] = (metrics['predictions'] == 1) & (self.targets == 0)
        metrics['FP'] = sum(metrics['FP_idx'])
        metrics['TN_idx'] = (metrics['predictions'] == 0) & (self.targets == 0)
        metrics['TN'] = sum(metrics['TN_idx'])
        metrics['FN_idx'] = (metrics['predictions'] == 0) & (self.targets == 1)
        metrics['FN'] = sum(metrics['FN_idx'])

        # ACC / Accuracy
        metrics['ACC'] = (metrics['TP'] + metrics['TN']) / len(self.targets)
        # TPR / True Positive Rate / Sensitivity / Recall
        metrics['TPR'] = np.float64(metrics['TP']) / (metrics['TP'] + metrics['FN'])
        # FNR / False Negative Rate
        metrics['FNR'] = np.float64(metrics['FN']) / (metrics['TP'] + metrics['FN'])
        # TNR / True Negative Rate / Specificity
        metrics['TNR'] = np.float64(metrics['TN']) / (metrics['FP'] + metrics['TN'])
        # FPR / False Positive Rate
        metrics['FPR'] = np.float64(metrics['FP']) / (metrics['FP'] + metrics['TN'])
        # PPV / Positive Predictive Value / Precision
        metrics['PPV'] = np.float64(metrics['TP']) / (metrics['TP'] + metrics['FP'])
        # NP / Negative Predictive Value
        metrics['NPV'] = np.float64(metrics['TN']) / (metrics['TN'] + metrics['FN'])
        
        assert (metrics['TPR'].astype(np.float16) == (1 - metrics['FNR']).astype(np.float16)) & \
               (metrics['TNR'].astype(np.float16) == (1 - metrics['FPR']).astype(np.float16))
        
        return metrics
    
    def confusion_table(self, path_out=None):

        df = pd.DataFrame(data=[[self.metrics['TP'], self.metrics['FP']], [self.metrics['FN'], self.metrics['TN']]],
                          index=['predicted_positive', 'predicted_negative'],
                          columns=['condition_positive', 'condition_negative'])

        if path_out is not None:
            if not os.path.exists(os.path.dirname(path_out)):
                os.makedirs(os.path.dirname(path_out))
            df.to_csv(path_out)

    def prediction_table(self, path_out):

        df = pd.DataFrame(index=self.sample_ind)
        df['targets'] = self.targets
        df['output_prob'] = np.round(self.output_prob, 2)
        df['predictions'] = self.predictions
        df['predictions_correct'] = self.predictions_correct
        df['true_positive'] = self.metrics['TP_idx']
        df['false_positive'] = self.metrics['FP_idx']
        df['true_negative'] = self.metrics['TN_idx']
        df['false_negative'] = self.metrics['FN_idx']

        if path_out is not None:
            if not os.path.exists(os.path.dirname(path_out)):
                os.makedirs(os.path.dirname(path_out))
            df.to_csv(path_out)
        
        
        
