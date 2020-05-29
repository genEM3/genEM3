import numpy as np
from torch import nn


def get_conv_pad(input_size, kernel_size, stride):
    padding = np.ceil(((stride-1)*input_size-stride+kernel_size)/2).astype(int)
    return padding


class AE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

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

