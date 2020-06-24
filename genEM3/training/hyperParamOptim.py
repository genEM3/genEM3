from ray import tune
import torch
from genEM3.model.autoencoder2d import AE, Encoder_4_sampling_bn, Decoder_4_sampling_bn
from genEM3.training.training import Trainer


class trainable(tune.Trainable):
    """Implementation of tune's hyper parameter optimization subclass"""
    def _setup(self, config):
        """Run when the experiment starts

        Args:
        config: dictionary of parameters used in the hyperparameter optimization.
                The keys of this dict are:lr, momentum, n_latent, n_fmaps

        """

        num_epoch_perTrain = 10,
        trainer_run_root = '/u/alik/tmpscratch/runlogs/AE_2d/hpOptim_test/',
        trainer_log_int = 128,
        trainer_save = True,
        device = 'cpu'
        # Model parameters
        input_size = 302
        output_size = input_size
        kernel_size = 3
        stride = 1
        n_fmaps = config.get("n_fmaps")
        n_latent = config.get("n_latent")
        cur_Model = AE(
            Encoder_4_sampling_bn(input_size, kernel_size, stride, n_fmaps, n_latent),
            Decoder_4_sampling_bn(output_size, kernel_size, stride, n_fmaps, n_latent))
        cur_Criterion = torch.nn.MSELoss()
        cur_Optimizer = torch.optim.SGD(cur_Model.parameters(), lr=config.get("lr"),
                                        momentum=config.get("momentum"))

        self.trainer = Trainer(run_root=trainer_run_root,
                               model=cur_Model,
                               optimizer=cur_Optimizer,
                               criterion=cur_Criterion,
                               train_loader=config.get("train_loader"),
                               validation_loader=config.get("validation_loader"),
                               num_epoch=num_epoch_perTrain,
                               log_int=trainer_log_int,
                               device=device,
                               save=trainer_save)

    def _train(self):

        """A single iteration of this method is run for each call.
        This should take more than a few econds and less than a few minutes"""
        self.Trainer.train()

    def _save(self):

        """ Saving the model """
        pass

    def _restore(self):

        """ Restoring the model"""
        pass

    def _stop(self):
        """This is run when the experiment ends"""
        pass
