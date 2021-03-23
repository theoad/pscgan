from torch import nn
import torch
import math
from torch.nn import functional as F

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, sampling='none'):
        super().__init__()
        self.sampling = sampling
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        # self.conv_func = F.conv_transpose2d if sampling == 'up' else F.conv2d
        self.stride = 2 if sampling == 'down' else 1
        self.padding = kernel_size // 2
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        if sampling == 'up':
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = None

    def forward(self, x, **kwargs):
        if self.up is not None:
            x = self.up(x)
        x = F.conv2d(x, self.weight * self.scale, bias=None, stride=self.stride, padding=self.padding)
        return x

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, '
                f'{self.weight.shape[2]}, stride={self.stride}, padding={self.padding})')


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sampling='none', activation=True, norm_type='none',
                 noise_injection_type='none'):
        super().__init__()
        layers = []
        if 'cat' in noise_injection_type:
            layers.append(noise_inject_factory[noise_injection_type](out_channels))

        layers.append(EqualConv2d(in_channels + int('cat' in noise_injection_type),
                                  out_channels, kernel_size, sampling=sampling))

        if norm_type != 'none':
            layers.append(norms_factory[norm_type](out_channels))

        if noise_injection_type in ['pcsf', 'ssf']:
            layers.append(noise_inject_factory[noise_injection_type](out_channels))
        elif noise_injection_type not in ['none', 'cat', 'cat_normalized']:
            raise Exception("Noise Injection Type Not Supported")

        self.layers = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(0.2) if activation else nn.Identity()

    def forward(self, x, **kwargs):
        for module in self.layers:
            x = module(x, **kwargs)
        return self.activation(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels, num_layers, upsample=True):
        super().__init__()
        if upsample:
            self.upsample = EqualConv2d(3, 3, 3, sampling='up')
        else:
            self.upsample = None
        self.conv = []
        self.conv.extend([ConvLayer(in_channels, in_channels, 3, norm_type='bias', noise_injection_type='none')] * (
                num_layers - 1))
        self.conv.append(ConvLayer(in_channels, 3, 1, activation=False, norm_type='bias', noise_injection_type='none'))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x, skip):
        out = self.conv(x)
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class NoiseInjectionCat(nn.Module):
    def forward(self, x, noise_std):
        batch, _, height, width = x.shape
        noise = x.new_empty(batch, 1, height, width).normal_() * noise_std
        return torch.cat((x, noise), dim=1)


class NoiseInjectionCatNormalized(nn.Module):
    def forward(self, x, noise_std):
        batch, _, height, width = x.shape
        noise = F.normalize(x.new_empty(batch, 1, height, width).normal_(), p=2, dim=(2, 3)) * noise_std
        return torch.cat((x, noise), dim=1)


class NoiseInjectionShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise_std):
        batch, _, height, width = x.shape
        noise = x.new_empty(batch, 1, height, width).normal_() * noise_std
        return x + self.weight * noise


class NoiseInjectionPerChannel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise_std):
        batch, _, height, width = x.shape
        noise = x.new_empty(batch, 1, height, width).normal_() * noise_std
        return x + self.weight * noise


noise_inject_factory = {
    'cat': lambda channels: NoiseInjectionCat(),
    'ssf': lambda channels: NoiseInjectionShared(),
    'pcsf': lambda channels: NoiseInjectionPerChannel(channels),
    'cat_normalized': lambda channels: NoiseInjectionCatNormalized()
}


class Bias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, **kwargs):
        return x + self.bias


norms_factory = {
    'instance': lambda channels: torch.nn.InstanceNorm2d(channels, affine=True),
    'batch': lambda channels: torch.nn.BatchNorm2d(channels, affine=True),
    'bias': lambda channels: Bias(channels)
}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm, sampling, noise_type, mid_cat_channels=0):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, 3, sampling=sampling if sampling == 'up' else 'none',
                               activation=True, norm_type=norm, noise_injection_type=noise_type)
        self.conv2 = ConvLayer(out_channels + mid_cat_channels, out_channels, 3,
                               sampling=sampling if sampling == 'down' else 'none', activation=True, norm_type=norm,
                               noise_injection_type=noise_type)

    def forward(self, x, mid_input, **kwargs):
        x = self.conv1(x, **kwargs)
        if mid_input is not None:
            x = torch.cat((x, mid_input), dim=1)
        x = self.conv2(x, **kwargs)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_layers, norm_type):
        layers = [ConvLayer(in_channels, out_channels, 3, activation=True, norm_type=norm_type)]
        layers.extend([ConvLayer(out_channels, out_channels, 3,
                                 activation=True, norm_type=norm_type)] * (num_layers - 1))
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, sampling):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3, norm_type='bias', sampling='none')
        self.conv2 = ConvLayer(in_channel, out_channel, 3, norm_type='bias', sampling=sampling)
        self.skip = ConvLayer(in_channel, out_channel, 1, sampling=sampling, activation=False, norm_type='none')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, activation=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim))

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, bias=self.bias)
        if self.activation:
            out = F.leaky_relu(out, negative_slope=0.2)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]},'
            f' {self.weight.shape[0]})'
        )


class ClassicResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3, norm_type='bias')
        self.conv2 = ConvLayer(in_channel, in_channel, 3, norm_type='bias')
        self.conv3 = ConvLayer(in_channel, out_channel, 3, norm_type='bias')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + x
