import numpy as np
import torch
import torch.optim as optim
# from torchvision import datasets

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

import os
import matplotlib.pyplot as plt

# make the log directory
logdir = "/gaba/u/alik/code/genEM3/playground/AK/raytune/__logs__/"

if not(os.path.isdir(logdir)):
    os.mkdir(logdir)


# Training loop
def train_mnist(config):
    model = ConvNet()
    train_loader, test_loader = get_data_loaders()
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        track.log(mean_accuracy=acc)
        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model, os.path.join(logdir, "model.pth"))


# Set-up experiment
search_space = {
    "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "momentum": tune.loguniform(0.8, 0.9999)
}

# /tmp is not accessible on GABA use the following dir:
ray.init(temp_dir='/tmpscratch/alik/runlogs/ray/')

analysis = tune.run(train_mnist,
                    config=search_space, num_samples=10,
                    resources_per_trial={'cpu': 4},
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
                    local_dir='/tmpscratch/alik/runlogs/ray_results')

# Test whether figures could be transferred using X11
fig, ax = plt.subplots()
dfs = analysis.trial_dataframes
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
plt.show()
