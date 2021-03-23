import torch
from torch import nn
from nets.utils import ToRGB, DoubleConv, ConvBlock


class EncDec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config['enc_cfg'])
        self.decoder = Decoder(config['dec_cfg'])

    def forward(self, y, noise_stds=0, encoder_assistance=True, **dec_kwargs):
        outputs = self.encoder(y)
        return self.decoder(outputs, encoder_assistance, noise_stds=noise_stds, **dec_kwargs)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm = config['norm']
        drips_norm = config['drips_norm']
        channels = config['channels']

        double_conv = DoubleConv(channels[0], channels[1], norm=norm, sampling='none', mid_cat_channels=0,
                                 noise_type='none')
        self.pipeline = [double_conv]
        self.drips = []

        last_channel = channels[1]
        for i, (drip_channel, channel, drip_depth) in enumerate(zip(config['drips_channels'],
                                                                    channels[2:],
                                                                    config['drips_depth'])):
            self.pipeline.append(DoubleConv(last_channel, channel, norm=norm,
                                            sampling='down', mid_cat_channels=0, noise_type='none'))
            self.drips.append(ConvBlock(last_channel, drip_channel, drip_depth, norm_type=drips_norm))
            last_channel = channel

        self.pipeline = nn.Sequential(*self.pipeline)
        self.drips = nn.Sequential(*self.drips)

    def forward(self, y):
        outputs = []
        for conv_net, drip in zip(self.pipeline[:-1], self.drips):
            y = conv_net(y, mid_input=None)
            outputs.append(drip(y))
        y = self.pipeline[-1](y, mid_input=None)
        # Return inputs from lowest scale to highest scale.
        return [y] + outputs[::-1]


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_trgb_layers = config['num_trgb_layers']
        norm = config['norm']
        channels = config['channels']

        first_conv_block = DoubleConv(channels[0], channels[0], norm=norm, sampling='none', mid_cat_channels=0,
                                      noise_type=config['noise_type'])
        first_trgb = ToRGB(channels[0], num_trgb_layers, upsample=False)

        self.conv_blocks = [first_conv_block]
        self.trgbs = [first_trgb]

        for drips_channel, in_channels, out_channels in zip(config['drips_channels'], channels[:-1], channels[1:]):
            conv_block = DoubleConv(in_channels, out_channels, norm=norm, sampling='up',
                                    mid_cat_channels=drips_channel, noise_type=config['noise_type'])
            to_rgb = ToRGB(out_channels, num_trgb_layers, upsample=True)
            self.conv_blocks.append(conv_block)
            self.trgbs.append(to_rgb)

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.trgbs = nn.Sequential(*self.trgbs)

    def forward(self, inputs, encoder_assistance, noise_stds=1):
        if type(noise_stds) is not list:
            noise_stds = [noise_stds] * len(self.conv_blocks)

        out = self.conv_blocks[0](inputs[0], mid_input=None, noise_std=noise_stds[0])
        trgb_out = self.trgbs[0](out, skip=None)
        for conv_block, trgb, residual, noise_std in zip(self.conv_blocks[1:], self.trgbs[1:], inputs[1:],
                                                         noise_stds[1:]):
            if not encoder_assistance:
                out = conv_block(out, mid_input=torch.zeros_like(residual), noise_std=noise_std)
            else:
                out = conv_block(out, mid_input=residual, noise_std=noise_std)
            trgb_out = trgb(out, skip=trgb_out)
        return trgb_out
