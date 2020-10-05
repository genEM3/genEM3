import torch
import torch.nn as nn
from torchsummary import summary

from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, Decoder_4_sampling_bn_1px_deep_convonly_skip


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Split_latent(nn.Module):
    # TODO: The goal is to split the latent space into two and use the split as mu and logvar.
    # The problem: this breaks down the symmetry between up and down sampling steps
    def forward(self, input):
        return torch.split(input, 2, dim=0)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(nn.Module):

    def __init__(self,
                 latent_size: int = 2048,
                 input_size: int = 140,
                 output_size: int = 140,
                 kernel_size: int = 3,
                 stride: int = 1,
                 batch_size: int = 256,
                 weight_KLD: float = 1.0):
        super().__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
                                    Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride, n_latent=latent_size), Flatten())

        # hidden => mui: 2048  is the number of latent variables fixed in the encoder/decoder models
        self.fc1 = nn.Linear(self.latent_size, self.latent_size)

        # hidden => logvar is a linear transformation of 2048 units to another 2048 units
        self.fc2 = nn.Linear(self.latent_size, self.latent_size)

        self.decoder = nn.Sequential(Unflatten(latent_size, 1, 1),
                                     Decoder_4_sampling_bn_1px_deep_convonly_skip(output_size, kernel_size, stride, n_latent=latent_size))

        self.cur_mu = torch.zeros([batch_size, self.latent_size], dtype=torch.float)
        self.cur_logvar = torch.zeros([batch_size, self.latent_size], dtype=torch.float)
        # The weight factor for the KL divergence part of loss. Currently set to 1
        self.weight_KLD = nn.Parameter(torch.Tensor([weight_KLD]), requires_grad=False)

    def encode(self, x):
        h = self.encoder(x)
        self.cur_mu, self.cur_logvar = self.fc1(h), self.fc2(h)

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self):
        if self.training:
            std = torch.exp(0.5 * self.cur_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.cur_mu)
        else:
            return self.cur_mu

    def forward(self, x):
        self.encode(x)
        z = self.reparameterize()
        return self.decode(z)

    def summary(self,
                device: str = "cpu",
                input_size: int = 140):
        summary(self, (1, input_size, input_size), device=device)
    
    @classmethod
    def from_saved_state_dict(cls,
                              model_dir: str = None, 
                              *args, 
                              **kwargs):
        """Initialize the model with the saved dictionary given with the model_dir"""
        model = cls(*args, **kwargs)
        checkpoint = torch.load(model_dir, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        return model