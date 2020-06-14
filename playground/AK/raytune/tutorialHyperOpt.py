import numpy as np
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

import os
import matplotlib.pyplot as plt

from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

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

space = {
    "lr": hp.loguniform("lr", 1e-10, 0.1),
    "momentum": hp.uniform("momentum", 0.1, 0.9),
}

hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

analysis = tune.run(train_mnist,
                    num_samples=10,
                    search_alg=hyperopt_search,
                    resources_per_trial={'cpu': 4},
                    local_dir='/tmpscratch/alik/runlogs/ray_results',
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"))
# Test whether figures could be transferred using X11
fig, ax = plt.subplots()
dfs = analysis.trial_dataframes
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
plt.show()
