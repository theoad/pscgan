import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from fractions import Fraction
import utils.utils as utils


class WGANBase(pl.LightningModule):
    def __init__(self, config, gen, disc):
        super().__init__()
        self.gen = gen
        self.disc = disc

    def compute_gp(self, real, fake, gen_input):
        batch_size = real.shape[0]
        num_channels = real.shape[1]
        patch_h = real.shape[2]
        patch_w = real.shape[3]
        # Gradient penalty calculation
        alpha = torch.rand((batch_size, 1), device=self.device)
        alpha = alpha.expand(batch_size, int(real.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, num_channels, patch_h, patch_w)

        interpolates = alpha * real.detach() + (1 - alpha) * fake.detach()
        interpolates.requires_grad_(True)

        disc_interpolates = self.disc(x=interpolates, y=gen_input)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.shape[0], -1)
        gradients_norm = gradients.norm(2, dim=1)
        return ((gradients_norm - 1) ** 2).mean()

    def gen_forward(self, gen_input, fake_require_grad, noise_stds=1):
        with torch.set_grad_enabled(fake_require_grad):
            fake = self.gen(y=gen_input, noise_stds=noise_stds, encoder_assistance=True)
        return fake


class DiscWGAN(WGANBase):
    def __init__(self, config, gen, disc):
        super().__init__(config, gen, disc)
        self.gp_reg = config['gp_reg']

    def __call__(self, real, gen_input):
        fake = self.gen_forward(gen_input, fake_require_grad=False)
        gp = self.compute_gp(real, fake, gen_input)

        d_out_fake_mean = torch.mean(self.disc(x=fake, y=gen_input))
        d_out_real_mean = torch.mean(self.disc(x=real, y=gen_input))
        minmax_loss = d_out_fake_mean - d_out_real_mean

        return minmax_loss + gp.mul_(self.gp_reg), {"disc_gp": gp, "disc_minmax": minmax_loss}


class GenExpectationLossWGAN(WGANBase):
    def __init__(self, config, gen, disc):
        super().__init__(config, gen, disc)
        self.minmax_reg = config['minmax_reg']
        self.expansion = config['expansion']
        self.penalty_frequency = config['penalty_frequency']
        self.penalty_batch_size = config['penalty_batch_size']

    def penalty_loss(self, real, gen_input):
        num_items_to_expand = self.penalty_batch_size
        mse_input = utils.expand_4d_batch(gen_input[:num_items_to_expand], self.expansion)
        fake = self.gen_forward(torch.cat((gen_input, mse_input), dim=0), True)
        restored_batch = utils.restore_expanded_4d_batch(fake[gen_input.shape[0]:], self.expansion).mean(0)
        mse = F.mse_loss(restored_batch, real[:num_items_to_expand])

        d_out_fake = self.disc(x=fake[:gen_input.shape[0]], y=gen_input)
        minmax = torch.neg(torch.mean(d_out_fake).mul_(self.minmax_reg))
        return mse + minmax, {"avg_sample_mse": mse, "gen_minmax": minmax}

    def minmax_loss(self, gen_input):
        fake = self.gen_forward(gen_input, True)
        d_out_fake = self.disc(x=fake, y=gen_input)
        minmax = torch.neg(torch.mean(d_out_fake).mul_(self.minmax_reg))
        return minmax, {"gen_minmax": minmax}

    def __call__(self, real, gen_input, batch_idx):
        if batch_idx % self.penalty_frequency == 0:
            return self.penalty_loss(real, gen_input)
        return self.minmax_loss(gen_input)


class GenLAGLoss(WGANBase):
    def __init__(self, config, gen, disc):
        super().__init__(config, gen, disc)
        self.minmax_reg = config['minmax_reg']

    def __call__(self, real, gen_input, **kwargs):
        x0 = self.gen_forward(gen_input, True, noise_stds=0)
        # latent_x0 = self.disc(x=x0, y=gen_input, latent_output=True)
        # latent_x = self.disc(x=real, y=gen_input, latent_output=True)
        mse = F.mse_loss(x0, real)
        d_out_fake = self.disc(x=self.gen_forward(gen_input, True, noise_stds=1), y=gen_input)
        minmax = torch.neg(torch.mean(d_out_fake).mul_(self.minmax_reg))
        logs = {"z0_mse": mse, "gen_minmax": minmax}
        return mse + minmax, logs


factory = {
    'disc_wgan': DiscWGAN,
    'gen_expectation_loss': GenExpectationLossWGAN,
    'gen_lag_loss': GenLAGLoss
}
