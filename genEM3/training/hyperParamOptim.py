from ray import tune


class trainable(tune.Trainable):
    """Implementation of tune's hyper parameter optimization subclass"""
    def _setup(self, config):

        """Run when the experiment starts"""
        pass

    def _train(self):

        """A single iteration of this method is run for each call. This should take more than a few seconds and less than a few minutes"""
        pass

    def _save(self):

        """ Saving the model """
        pass

    def _restore(self):
        
        """ Restoring the model"""
        pass

    def _stop(self):
        """This is run when the experiment ends"""
        pass
