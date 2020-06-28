from ray import tune
import torch
import os
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

        num_epoch_perTrain = 1
        trainer_run_root = '/u/alik/tmpscratch/runlogs/AE_2d/hpOptim_test/'
        # create directory if not preset
        if not(os.path.isdir(trainer_run_root)):
            os.mkdir(trainer_run_root)
        trainer_log_int = 128
        trainer_save = True
        device = 'cuda'
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
        val_loss_dict = self.trainer.train()
        return val_loss_dict
    
    def _save(self, tmp_checkpoint_dir):

        """ Saving the model """
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.trainer.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):

        """ Restoring the model"""
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.trainer.model.load_state_dict(torch.load(checkpoint_path))

# If necessary in the future implement stop
#     def _stop(self):
#         """This is run when the experiment ends"""
#         pass
