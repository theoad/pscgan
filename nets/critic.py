from torch import nn
from nets.utils import ConvLayer, ResBlock, EqualLinear, ClassicResBlock
import torch
from nets.encdec import Encoder


class BasicCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config['channels'][:-1]
        self.out_channels = config['channels'][1:]
        self.f_rgb = ConvLayer(3, self.in_channels[0], 1, norm_type='bias')
        self.blocks = []
        self.init_channels()
        for index, (in_chan, out_chan) in enumerate(zip(self.in_channels,
                                                        self.out_channels)):
            block = ResBlock(in_chan, out_chan, 'down')
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

        self.final_conv = ConvLayer(self.out_channels[-1], self.out_channels[-1], 3, norm_type='bias')
        out_spatial_extent = config['input_spatial_extent'] // 2 ** (len(self.out_channels))
        in_channels = (out_spatial_extent ** 2) * self.out_channels[-1]
        self.linear = nn.Sequential(EqualLinear(in_channels, self.out_channels[-1], activation=True),
                                    EqualLinear(self.out_channels[-1], 1, activation=False))

    def init_channels(self):
        pass

    def forward(self, x, **kwargs):
        out = self.f_rgb(x)
        out = self.blocks(out)
        out = self.final_conv(out).view(x.shape[0], -1)
        return self.linear(out)


class YPreProcessCritic(BasicCritic):
    def __init__(self, config):
        super().__init__(config)
        self.y_conv = nn.Sequential(
            ConvLayer(3, 64, 3, norm_type='bias'),
            ClassicResBlock(64, 64),
            ClassicResBlock(64, 64),
            ClassicResBlock(64, 64),
            ConvLayer(64, 64, 3, norm_type='bias')
        )

    def init_channels(self):
        self.in_channels[0] += 64

    def forward(self, x, **kwargs):
        out = torch.cat((self.f_rgb(x), self.y_conv(kwargs['y'])), dim=1)
        out = self.blocks(out)
        out = self.final_conv(out).view(x.shape[0], -1)
        return self.linear(out)


class YPreProcessCriticOld(YPreProcessCritic):
    def __init__(self, config):
        super().__init__(config)
        self.y_conv = nn.Sequential(
            ConvLayer(3, 64, 3, norm_type='bias'),
            ConvLayer(64, 64, 3, norm_type='bias'),
            ConvLayer(64, 64, 3, norm_type='bias'),
            ConvLayer(64, 64, 3, norm_type='bias'),
        )


class YPreProcessCriticOldLAG(YPreProcessCriticOld):
    def forward(self, x, latent_output=False, **kwargs):
        out = torch.cat((self.f_rgb(x), self.y_conv(kwargs['y'])), dim=1)
        out = self.blocks(out)
        out = self.final_conv(out).view(x.shape[0], -1)
        if latent_output:
            return out
        return self.linear(out)


class YPostProcessCritic(BasicCritic):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config['enc_cfg'])

    def init_channels(self):
        for i, c in enumerate(self.config['enc_cfg']['drips_channels'] + [self.config['enc_cfg']['channels'][-1]]):
            self.in_channels[i] += c

    def forward(self, x, **kwargs):
        enc_result = self.encoder(kwargs['y'])
        enc_result.reverse()
        out = self.f_rgb(x)
        for res_block, enc_out in zip(self.blocks, enc_result):
            out = res_block(torch.cat((enc_out, out), dim=1))

        for res_block in self.blocks[len(enc_result):]:
            out = res_block(out)
        out = self.final_conv(out).view(x.shape[0], -1)
        return self.linear(out)
