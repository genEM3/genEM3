import torch
import torch.nn as nn

from genEM3.model.autoencoder2d import Encoder_4_sampling_bn_1px_deep_convonly_skip, Decoder_4_sampling_bn_1px_deep_convonly_skip

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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
                 stride: int = 1):
        super().__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(Encoder_4_sampling_bn_1px_deep_convonly_skip(input_size, kernel_size, stride), Flatten())

        # hidden => mui: 2048  is the number of latent variables fixed in the encoder/decoder models
        self.fc1 = nn.Linear(2048, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(2048, self.latent_size)

        self.decoder = nn.Sequential(Unflatten(latent_size, 1, 1), 
                                     Decoder_4_sampling_bn_1px_deep_convonly_skip(output_size, kernel_size, stride),
                                     nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
