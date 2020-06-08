import ray
from ray import tune

# /tmp is not accessible on GABA use the following dir:
ray.init(temp_dir='/tmpscratch/alik/runlogs/ray/')


# Objective function that is measured
def objective(x, a, b):
    return a * (x ** 0.5) + b


# function based API example
def trainable(config):
    # config (dict): A dict of hyperparameters.

    for x in range(20):
        score = objective(x, config["a"], config["b"])

        tune.track.log(score=score)  # This sends the score to Tune.


# class based API example
class Trainable(tune.Trainable):
    def _setup(self, config):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]

    def _train(self):  # This is called iteratively.
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}


# Run with a and b uniformly sampled from (-1,1)
space = {"a": tune.uniform(-1, 1), "b": tune.uniform(-1, 1)}
tune.run(trainable, config=space, num_samples=100)
