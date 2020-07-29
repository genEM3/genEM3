import numpy as np
import torch
from torch import nn


def get_conv_pad(input_size, kernel_size, stride):
    padding = np.ceil(((stride-1)*input_size-stride+kernel_size)/2).astype(int)
    return padding


class AE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.iteration = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.epoch = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Encoder_4_sampling_1px_deep_convonly_skip(nn.Module):

    def __init__(self, input_size, kernel_size, stride, n_fmaps=16, n_latent=2048):
        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.max_pool2 = nn.MaxPool2d(2)

        self.encoding_conv11 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv12 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv21 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps * 2, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv22 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv31 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 4, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv32 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 6, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv41 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 8, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv42 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 12, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv51 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 16, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_conv52 = nn.Sequential(
            nn.Conv2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.LeakyReLU())

        self.encoding_latent = nn.Sequential(
            nn.Conv2d(n_fmaps * 16 + 6400 + 25088, n_latent, 1, stride),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.encoding_conv11(x)
        x = self.encoding_conv12(x)
        x = self.max_pool2(x)
        x = self.encoding_conv21(x)
        x = self.encoding_conv22(x)
        x = self.max_pool2(x)
        x = self.encoding_conv31(x)
        x = self.encoding_conv32(x)
        x_split = torch.split(x, int(x.shape[1]/3*2), dim=1)
        x = x_split[0]
        x_skip_3 = x_split[1].reshape((x_split[1].shape[0], -1, 1, 1))
        x = self.max_pool2(x)
        x = self.encoding_conv41(x)
        x = self.encoding_conv42(x)
        x_split = torch.split(x, int(x.shape[1]/3*2), dim=1)
        x = x_split[0]
        x_skip_4 = x_split[1].reshape((x_split[1].shape[0], -1, 1, 1))
        x = self.max_pool2(x)
        x = self.encoding_conv51(x)
        x = self.encoding_conv52(x)
        x_latent_input = torch.cat((x, x_skip_4, x_skip_3), dim=1)
        x = self.encoding_latent(x_latent_input)

        return x


class Decoder_4_sampling_1px_deep_convonly_skip(nn.Module):

    def __init__(self, output_size, kernel_size, stride, n_fmaps=16, n_latent=2048):
        super().__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.decoding_latent = nn.Sequential(
            nn.ConvTranspose2d(n_latent, n_fmaps * 16 + 6400 + 25088, 1, stride),
            nn.LeakyReLU())

        self.decoding_convt52 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt51 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 8, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt42 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 12, n_fmaps * 8, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt41 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 4, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt32 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 6, n_fmaps * 4, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt31 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 2, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt22 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt21 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 1, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt12 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.LeakyReLU())

        self.decoding_convt11 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))

    def forward(self, x):

        x_latent_output = self.decoding_latent(x)

        x = x_latent_output[:, :self.n_fmaps * 16, :, :]
        x_skip_4 = x_latent_output[:, self.n_fmaps * 16:self.n_fmaps * 16 + 6400, :, :]
        x_skip_3 = x_latent_output[:, self.n_fmaps * 16 + 6400::, :, :]

        x = self.decoding_convt52(x)
        x = self.decoding_convt51(x)
        x = self.up_sample2(x)
        x = torch.cat((x, x_skip_4.reshape((x_skip_4.shape[0], 64, 10, 10))), dim=1)
        x = self.decoding_convt42(x)
        x = self.decoding_convt41(x)
        x = self.up_sample2(x)
        x = torch.cat((x, x_skip_3.reshape((x_skip_3.shape[0], 32, 28, 28))), dim=1)
        x = self.decoding_convt32(x)
        x = self.decoding_convt31(x)
        x = self.up_sample2(x)
        x = self.decoding_convt22(x)
        x = self.decoding_convt21(x)
        x = self.up_sample2(x)
        x = self.decoding_convt12(x)
        x = self.decoding_convt11(x)

        return x

class Encoder_4_sampling_bn_1px_deep_convonly_skip(nn.Module):

    def __init__(self, input_size, kernel_size, stride, n_fmaps=16, n_latent=2048):
        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.max_pool2 = nn.MaxPool2d(2)

        self.encoding_conv11 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.encoding_conv12 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.encoding_conv21 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.encoding_conv22 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.encoding_conv31 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.encoding_conv32 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 6, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 6),
            nn.LeakyReLU())

        self.encoding_conv41 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.encoding_conv42 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 12, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 12),
            nn.LeakyReLU())

        self.encoding_conv51 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.encoding_conv52 = nn.Sequential(
            nn.Conv2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.encoding_latent = nn.Sequential(
            nn.Conv2d(n_fmaps * 16 + 6400 + 25088, n_latent, 1, stride),
            nn.BatchNorm2d(n_latent),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.encoding_conv11(x)
        x = self.encoding_conv12(x)
        x = self.max_pool2(x)
        x = self.encoding_conv21(x)
        x = self.encoding_conv22(x)
        x = self.max_pool2(x)
        x = self.encoding_conv31(x)
        x = self.encoding_conv32(x)
        x_split = torch.split(x, int(x.shape[1]/3*2), dim=1)
        x = x_split[0]
        x_skip_3 = x_split[1].reshape((x_split[1].shape[0], -1, 1, 1))
        x = self.max_pool2(x)
        x = self.encoding_conv41(x)
        x = self.encoding_conv42(x)
        x_split = torch.split(x, int(x.shape[1]/3*2), dim=1)
        x = x_split[0]
        x_skip_4 = x_split[1].reshape((x_split[1].shape[0], -1, 1, 1))
        x = self.max_pool2(x)
        x = self.encoding_conv51(x)
        x = self.encoding_conv52(x)
        x_latent_input = torch.cat((x, x_skip_4, x_skip_3), dim=1)
        x = self.encoding_latent(x_latent_input)

        return x


class Decoder_4_sampling_bn_1px_deep_convonly_skip(nn.Module):

    def __init__(self, output_size, kernel_size, stride, n_fmaps=16, n_latent=2048):
        super().__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.decoding_latent = nn.Sequential(
            nn.ConvTranspose2d(n_latent, n_fmaps * 16 + 6400 + 25088, 1, stride),
            nn.BatchNorm2d(n_fmaps * 16 + 6400 + 25088),
            nn.LeakyReLU())

        self.decoding_convt52 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.decoding_convt51 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.decoding_convt42 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 12, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.decoding_convt41 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.decoding_convt32 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 6, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.decoding_convt31 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.decoding_convt22 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.decoding_convt21 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 1, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 1),
            nn.LeakyReLU())

        self.decoding_convt12 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.decoding_convt11 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))

    def forward(self, x):

        x_latent_output = self.decoding_latent(x)

        x = x_latent_output[:, :self.n_fmaps * 16, :, :]
        x_skip_4 = x_latent_output[:, self.n_fmaps * 16:self.n_fmaps * 16 + 6400, :, :]
        x_skip_3 = x_latent_output[:, self.n_fmaps * 16 + 6400::, :, :]

        x = self.decoding_convt52(x)
        x = self.decoding_convt51(x)
        x = self.up_sample2(x)
        x = torch.cat((x, x_skip_4.reshape((x_skip_4.shape[0], 64, 10, 10))), dim=1)
        x = self.decoding_convt42(x)
        x = self.decoding_convt41(x)
        x = self.up_sample2(x)
        x = torch.cat((x, x_skip_3.reshape((x_skip_3.shape[0], 32, 28, 28))), dim=1)
        x = self.decoding_convt32(x)
        x = self.decoding_convt31(x)
        x = self.up_sample2(x)
        x = self.decoding_convt22(x)
        x = self.decoding_convt21(x)
        x = self.up_sample2(x)
        x = self.decoding_convt12(x)
        x = self.decoding_convt11(x)

        return x

class Encoder_4_sampling_bn_1px_deep_convonly(nn.Module):

    def __init__(self, input_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.encoding_conv11 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.encoding_conv12 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv21 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.encoding_conv22 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv31 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.encoding_conv32 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv41 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.encoding_conv42 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv51 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.encoding_conv52 = nn.Sequential(
            nn.Conv2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.encoding_latent = nn.Sequential(
            nn.Conv2d(n_fmaps * 16, n_latent, 1, stride),
            nn.BatchNorm2d(n_latent),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.encoding_conv11(x)
        x = self.encoding_conv12(x)
        x = self.encoding_conv21(x)
        x = self.encoding_conv22(x)
        x = self.encoding_conv31(x)
        x = self.encoding_conv32(x)
        x = self.encoding_conv41(x)
        x = self.encoding_conv42(x)
        x = self.encoding_conv51(x)
        x = self.encoding_conv52(x)
        x = self.encoding_latent(x)

        return x


class Decoder_4_sampling_bn_1px_deep_convonly(nn.Module):

    def __init__(self, output_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.decoding_latent = nn.Sequential(
            nn.ConvTranspose2d(n_latent, n_fmaps * 16, 1, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.decoding_convt52 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.decoding_convt51 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.decoding_convt42 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.decoding_convt41 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.decoding_convt32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.decoding_convt31 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.decoding_convt22 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.decoding_convt21 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 1, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 1),
            nn.LeakyReLU())

        self.decoding_convt12 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.decoding_convt11 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))

    def forward(self, x):
        x = self.decoding_latent(x)
        x = self.decoding_convt52(x)
        x = self.decoding_convt51(x)
        x = self.decoding_convt42(x)
        x = self.decoding_convt41(x)
        x = self.decoding_convt32(x)
        x = self.decoding_convt31(x)
        x = self.decoding_convt22(x)
        x = self.decoding_convt21(x)
        x = self.decoding_convt12(x)
        x = self.decoding_convt11(x)

        return x

class Encoder_4_sampling_bn_1px_deep(nn.Module):

    def __init__(self, input_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.encoding_conv11 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.encoding_conv12 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv21 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.encoding_conv22 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv31 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.encoding_conv32 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv41 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.encoding_conv42 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv51 = nn.Sequential(
            nn.Conv2d(n_fmaps * 8, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.encoding_conv52 = nn.Sequential(
            nn.Conv2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.encoding_fc = nn.Sequential(
            nn.Linear(1 ** 2 * n_fmaps * 16, n_latent),
            nn.LeakyReLU())

    def forward(self, x):
        x = self.encoding_conv11(x)
        x = self.encoding_conv12(x)
        x = self.encoding_conv21(x)
        x = self.encoding_conv22(x)
        x = self.encoding_conv31(x)
        x = self.encoding_conv32(x)
        x = self.encoding_conv41(x)
        x = self.encoding_conv42(x)
        x = self.encoding_conv51(x)
        x = self.encoding_conv52(x)
        x = self.encoding_fc(x.reshape((-1, 1, 1 ** 2 * self.n_fmaps * 16)))

        return x


class Decoder_4_sampling_bn_1px_deep(nn.Module):

    def __init__(self, output_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.decoding_fc = nn.Sequential(
            nn.Linear(n_latent, 1 ** 2 * n_fmaps * 16),
            nn.LeakyReLU())

        self.decoding_convt52 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 16, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 16),
            nn.LeakyReLU())

        self.decoding_convt51 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 16, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.decoding_convt42 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.LeakyReLU())

        self.decoding_convt41 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.decoding_convt32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.LeakyReLU())

        self.decoding_convt31 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.decoding_convt22 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.LeakyReLU())

        self.decoding_convt21 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps * 1, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 1),
            nn.LeakyReLU())

        self.decoding_convt12 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.LeakyReLU())

        self.decoding_convt11 = nn.Sequential(
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))

    def forward(self, x):
        x = self.decoding_fc(x)
        x = self.decoding_convt52(x.reshape((-1, self.n_fmaps * 16, 1, 1)))
        x = self.decoding_convt51(x)
        x = self.decoding_convt42(x)
        x = self.decoding_convt41(x)
        x = self.decoding_convt32(x)
        x = self.decoding_convt31(x)
        x = self.decoding_convt22(x)
        x = self.decoding_convt21(x)
        x = self.decoding_convt12(x)
        x = self.decoding_convt11(x)

        return x


class Encoder_4_sampling_bn_1px(nn.Module):

    def __init__(self, input_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.encoding_conv1 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv2 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv3 = nn.Sequential(
            nn.Conv2d(n_fmaps * 2, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.encoding_conv4 = nn.Sequential(
            nn.Conv2d(n_fmaps * 4, n_fmaps * 8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 8),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.encoding_fc = nn.Sequential(
            nn.Linear(1 ** 2 * n_fmaps * 8, n_latent),
            nn.ReLU())

    def forward(self, x):
        x = self.encoding_conv1(x)
        x = self.encoding_conv2(x)
        x = self.encoding_conv3(x)
        x = self.encoding_conv4(x)
        x = self.encoding_fc(x.reshape((-1, 1, 1 ** 2 * self.n_fmaps * 8)))

        return x


class Decoder_4_sampling_bn_1px(nn.Module):

    def __init__(self, output_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent

        self.decoding_fc = nn.Sequential(
            nn.Linear(n_latent, 1 ** 2 * n_fmaps * 8),
            nn.ReLU())

        self.decoding_convt1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 8, n_fmaps * 4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 4),
            nn.ReLU())

        self.decoding_convt2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 4, n_fmaps * 2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps * 2),
            nn.ReLU())

        self.decoding_convt3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps * 2, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU())

        self.decoding_convt4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))

    def forward(self, x):
        x = self.decoding_fc(x)
        x = self.decoding_convt1(x.reshape((-1, self.n_fmaps * 8, 1, 1)))
        x = self.decoding_convt2(x)
        x = self.decoding_convt3(x)
        x = self.decoding_convt4(x)

        return x


class Encoder_4_sampling_bn(nn.Module):
    
    def __init__(self, input_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()
        
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent
        
        self.encoding_conv1 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(2))
    
        self.encoding_conv2 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps*2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.encoding_conv3 = nn.Sequential(
            nn.Conv2d(n_fmaps*2, n_fmaps*4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.encoding_conv4 = nn.Sequential(
            nn.Conv2d(n_fmaps*4, n_fmaps*8, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.encoding_fc = nn.Sequential(
            nn.Linear(17**2*n_fmaps*8, n_latent),
            nn.ReLU())
        
    def forward(self, x):
        
        x = self.encoding_conv1(x)
        x = self.encoding_conv2(x)
        x = self.encoding_conv3(x)
        x = self.encoding_conv4(x)
        x = self.encoding_fc(x.reshape((-1, 1, 17**2*self.n_fmaps*8)))
        
        return x


class Decoder_4_sampling_bn(nn.Module):
    
    def __init__(self, output_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()
        
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent
        
        self.decoding_fc = nn.Sequential(
            nn.Linear(n_latent, 17**2*n_fmaps*8),
            nn.ReLU())
        
        self.decoding_convt1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps*8, n_fmaps*4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*4),
            nn.ReLU())
        
        self.decoding_convt2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps*4, n_fmaps*2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU())

        self.decoding_convt3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps*2, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU())
        
        self.decoding_convt4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))
        
    def forward(self, x):
        
        x = self.decoding_fc(x)
        x = self.decoding_convt1(x.reshape((-1, self.n_fmaps*8, 17, 17)))
        x = self.decoding_convt2(x)
        x = self.decoding_convt3(x)
        x = self.decoding_convt4(x)
        
        return x


class Encoder_3_sampling_bn(nn.Module):
    
    def __init__(self, input_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()
        
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent
        
        self.encoding_conv1 = nn.Sequential(
            nn.Conv2d(1, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU(),
            nn.MaxPool2d(2))
    
        self.encoding_conv2 = nn.Sequential(
            nn.Conv2d(n_fmaps, n_fmaps*2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.encoding_conv3 = nn.Sequential(
            nn.Conv2d(n_fmaps*2, n_fmaps*4, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.encoding_fc = nn.Sequential(
            nn.Linear(36**2*n_fmaps*4, n_latent),
            nn.ReLU())
        
    def forward(self, x):
        
        x = self.encoding_conv1(x)
        x = self.encoding_conv2(x)
        x = self.encoding_conv3(x)
        x = self.encoding_fc(x.reshape((-1, 1, 36**2*self.n_fmaps*4)))
        
        return x


class Decoder_3_sampling_bn(nn.Module):
    
    def __init__(self, output_size, kernel_size, stride, n_fmaps, n_latent):
        super().__init__()
        
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fmaps = n_fmaps
        self.n_latent = n_latent
        
        self.decoding_fc = nn.Sequential(
            nn.Linear(n_latent, 36**2*n_fmaps*4),
            nn.ReLU())
        
        self.decoding_convt1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps*4, n_fmaps*2, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU())
        
        self.decoding_convt2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps*2, n_fmaps, kernel_size, stride),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU())
        
        self.decoding_convt3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(n_fmaps, 1, kernel_size, stride))
        
    def forward(self, x):
        
        x = self.decoding_fc(x)
        x = self.decoding_convt1(x.reshape((-1, self.n_fmaps*4, 36, 36)))
        x = self.decoding_convt2(x)
        x = self.decoding_convt3(x)
        
        return x

