import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

class LatentSampler:

    def __init__(self,
                 run_root,
                 model,
                 dataloader):

        self.run_root = run_root
        self.model = model
        self.dataloader = dataloader
        self.X = []
        self.y = []

    def sample(self):

        for i, data in enumerate(self.dataloader):
            inputs = data['input']
            targets = data['target']
            outputs = self.model.encode_input(inputs)

            if i == 0:
                targets_all = targets.data.numpy()
                outputs_all = outputs.data.numpy()
            else:
                targets_all = np.concatenate([targets_all, targets.data.numpy()], axis=0)
                outputs_all = np.concatenate([outputs_all, outputs.data.numpy()], axis=0)

        self.X = outputs_all.squeeze()
        self.y = targets_all.squeeze()

    def pca(self, n_components, plot):

        pca = decomposition.PCA(n_components=n_components)
        pca.fit(self.X)
        XPC = pca.transform(self.X)

        if plot:
            fig = plt.figure(1, figsize=(4, 3))
            plt.clf()
            if n_components == 2:
                ax = plt.axes()
                ax.scatter(XPC[:, 0], XPC[:, 1], c=self.y, cmap=plt.cm.Set1)
            elif n_components == 3:
                ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
                ax.scatter(XPC[:, 0], XPC[:, 1], XPC[:, 2], c=self.y, cmap=plt.cm.Set1)
            else:
                print('Can only plot for n_components == 2 or 3')

