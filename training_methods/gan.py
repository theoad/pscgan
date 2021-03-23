import pytorch_lightning as pl
from nets.factory import factory as nets_fac
from optimization.loss_functions import factory as loss_fac
from optimization.optimizers import factory as opt_fac
import numpy as np
from datetime import datetime
import os
import torch
import torchvision.utils as vutils
from pytorch_lightning.metrics import PSNR
import matplotlib.pyplot as plt
import utils.utils as utils
import math


class GAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gen = nets_fac[config['gen_cfg']['type']](config['gen_cfg'])
        self.disc = nets_fac[config['disc_cfg']['type']](config['disc_cfg'])
        self.gen_loss = loss_fac[config['gen_cfg']['loss_cfg']['type']](config['gen_cfg']['loss_cfg'],
                                                                        self.gen, self.disc)
        self.disc_loss = loss_fac[config['disc_cfg']['loss_cfg']['type']](config['disc_cfg']['loss_cfg'],
                                                                          self.gen, self.disc)
        self.num_disc_steps = config['num_disc_steps']
        if 'test_cfg' in config:
            self.test_cfg = config['test_cfg']
            self.noise_std_traversal = config['test_cfg']['noise_std_traversal']
            self.num_avg_samples_traversal = config['test_cfg']['num_avg_samples_traversal']
            self.num_fid_evals = config['test_cfg']['num_fid_evals']
            self.divide_expanded_forward_pass = config['test_cfg']['divide_expanded_forward_pass']
        self.collages = None
        self.ours_s_fids = None
        self.ours_a_fids = None
        self.psnr_for_ours_a_fid = None
        self.psnr_for_ours_s_fid = None
        self.collage_metric = None
        self.val_path = None
        self.m_real = None
        self.s_real = None
        self.test_path = None
        self.denoiser_criteria = None

    def on_load_checkpoint(self, checkpoint):
        sd = self.state_dict()
        for param in sd:
            if param in checkpoint['state_dict'] and sd[param].size() != checkpoint['state_dict'][param].size():
                del checkpoint['state_dict'][param]

    def configure_optimizers(self):
        gen_opt = opt_fac[self.config['optim_cfg']['type']](self.gen.parameters(), self.config['optim_cfg'])
        disc_opt = opt_fac[self.config['optim_cfg']['type']](self.disc.parameters(), self.config['optim_cfg'])
        return {'optimizer': gen_opt, 'frequency': 1}, {'optimizer': disc_opt, 'frequency': self.num_disc_steps}

    def forward(self, y, **kwargs):
        gen_out = self.gen(y=y, encoder_assistance=True, **kwargs)
        return gen_out

    def batch_postprocess(self, batch):
        return batch['real'], batch['noisy']

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = self.batch_postprocess(batch)
        if optimizer_idx == 0:
            loss, logs = self.gen_loss(real=x, gen_input=y, batch_idx=batch_idx)
            self.log_dict(logs, prog_bar=True, logger=True)
        else:
            loss, logs = self.disc_loss(real=x, gen_input=y)
            self.log_dict(logs, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.collage_metric is None:
            self.collage_metric = utils.CollageVal().to(self.device)
        x, y = self.batch_postprocess(batch)
        y_expanded = utils.expand_4d_batch(y, 4)

        with torch.no_grad():
            out = self(y=y_expanded, noise_stds=1)
        if batch_idx == 0:
            self.collage_metric.update(x)
            self.collage_metric.update(out)

    def validation_epoch_end(self, outputs):
        out = self.collage_metric.compute()
        self.collage_metric.reset()
        fig = plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(vutils.make_grid(out.clamp_(0, 1).detach().cpu(), padding=2,
                                                 normalize=False, range=(0, 1)), (1, 2, 0)))
        fig.savefig(os.path.join(self.val_path, str(self.current_epoch).zfill(5) + "_fake_collage_" +
                                 datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p") + ".png"), dpi=350)

    def on_test_epoch_start(self):
        if self.test_cfg['collages']:
            self.collages = torch.nn.ModuleDict(
                {str(i): utils.Collage(i, self.test_path, 8,
                                       ["real", "noisy"] +
                                       ['fake_z' + str(noise_std) for noise_std in self.noise_std_traversal] +
                                       ["mean", "std_dev_z1"]).to(self.device) for i in self.test_cfg['save_batch']})

        if self.test_cfg['fid_and_psnr']:
            self.m_real, self.s_real, model = utils.init_fid(self.test_cfg['training_data_stats_path'],
                                                             self.train_dataloader(),
                                                             self.device,
                                                             verbose=False)

            self.ours_a_fids = torch.nn.ModuleList([utils.FID(1, self.m_real, self.s_real, model)
                                                    for _ in self.num_avg_samples_traversal]).to(self.device)
            self.psnr_for_ours_a_fid = torch.nn.ModuleList([PSNR(1)
                                                            for _ in self.num_avg_samples_traversal]).to(self.device)
            self.ours_s_fids = torch.nn.ModuleList([utils.FID(self.num_fid_evals, self.m_real, self.s_real, model)
                                                    for _ in self.noise_std_traversal]).to(self.device)
            self.psnr_for_ours_s_fid = torch.nn.ModuleList([PSNR(1)
                                                            for _ in self.noise_std_traversal]).to(self.device)
        if self.test_cfg['denoiser_criteria']:
            avg_kernel = 1/(3*15*15) * torch.ones(1, 3, 15, 15).to(self.device)
            self.denoiser_criteria = utils.DenoiserCriteria(avg_kernel).to(self.device)

    def forward_with_divisor(self, y, divisor, **kwargs):
        out = []
        for i in range(divisor):
            out.append(self(y[i * y.shape[0] // divisor: (i + 1) * y.shape[0] // divisor], **kwargs))
        return torch.cat(out, dim=0)

    def test_step(self, batch, batch_idx):
        x, y = self.batch_postprocess(batch)
        with torch.no_grad():
            save_collage = self.test_cfg['collages'] and batch_idx in self.test_cfg['save_batch']
            idx = None
            if self.test_cfg['fid_and_psnr'] or save_collage:
                if save_collage:
                    idx = str(batch_idx)
                    self.collages[idx].set_batch_size(x.shape[0])
                    self.collages[idx].update("real", x)
                    self.collages[idx].update("noisy", y)
                expansion = max(8, self.num_fid_evals) if save_collage else self.num_fid_evals
                y_expanded = utils.expand_4d_batch(y, expansion)
                x_expanded = utils.expand_4d_batch(x, expansion)
                out_reshaped_64_sigma_1 = None
                for i, noise_stds in enumerate(self.noise_std_traversal):
                    out = self.forward_with_divisor(y_expanded, self.divide_expanded_forward_pass,
                                                    noise_stds=noise_stds)
                    out_reshaped = utils.restore_expanded_4d_batch(out, expansion)
                    if self.test_cfg['fid_and_psnr']:
                        self.ours_s_fids[i].update(out_reshaped[:self.num_fid_evals])
                        self.psnr_for_ours_s_fid[i].update(x_expanded, out)
                    if save_collage:
                        self.collages[idx].update("fake_z" + str(noise_stds), out_reshaped[:8])
                        if expansion == 64 and noise_stds == 1:
                            out_reshaped_64_sigma_1 = out_reshaped
                if self.test_cfg['fid_and_psnr']:
                    for i, ours_a_expansion in enumerate(self.num_avg_samples_traversal):
                        out = self.forward_with_divisor(utils.expand_4d_batch(y, ours_a_expansion),
                                                        self.divide_expanded_forward_pass, noise_stds=1)
                        out_reshaped_fid = utils.restore_expanded_4d_batch(out, ours_a_expansion)
                        out_fid_mean = out_reshaped_fid.mean(0)
                        self.ours_a_fids[i].update(out_fid_mean.unsqueeze(0))
                        self.psnr_for_ours_a_fid[i].update(x, out_fid_mean)
                        if save_collage and ours_a_expansion == 64:
                            out_reshaped_64_sigma_1 = out_reshaped_fid

                if save_collage:
                    if out_reshaped_64_sigma_1 is None:
                        out = self.forward_with_divisor(utils.expand_4d_batch(y, 64),
                                                        self.divide_expanded_forward_pass, noise_stds=1)
                        out_reshaped_64_sigma_1 = utils.restore_expanded_4d_batch(out, 64)
                    self.collages[idx].update("mean", out_reshaped_64_sigma_1.mean(0))
                    self.collages[idx].update("std_dev_z1", out_reshaped_64_sigma_1.std(0) ** (1 / 4))

            if self.test_cfg['denoiser_criteria']:
                out = self.forward_with_divisor(y, 1, noise_stds=1)
                self.denoiser_criteria.update(out - x, y - out, y - x, self.device)

    def test_epoch_end(self, outputs):
        if self.test_cfg['fid_and_psnr']:
            for i, noise_stds in enumerate(self.noise_std_traversal):
                ours_s_fid_scores = self.ours_s_fids[i].compute()
                self.log("Sigma_z=" + str(noise_stds) + "_FID_mean", torch.mean(ours_s_fid_scores), prog_bar=True,
                         logger=True)
                self.log("Sigma_z=" + str(noise_stds) + "_FID_std", torch.std(ours_s_fid_scores), prog_bar=True,
                         logger=True)
                self.log("Sigma_z=" + str(noise_stds) + "_PSNR", self.psnr_for_ours_s_fid[i].compute(), prog_bar=True,
                         logger=True)
            for i, num_expansions in enumerate(self.num_avg_samples_traversal):
                self.log("N=" + str(num_expansions) + "_FID", self.ours_a_fids[i].compute(), prog_bar=True, logger=True)
                self.log("N=" + str(num_expansions) + "_PSNR", self.psnr_for_ours_a_fid[i].compute(), prog_bar=True,
                         logger=True)

        if self.test_cfg['collages']:
            for idx in self.collages:
                zfill = max(self.test_cfg['save_batch'])
                self.collages[idx].compute(math.ceil(math.log10(zfill)))

        if self.test_cfg['denoiser_criteria']:
            save_path = os.path.join(self.test_path, "histograms")
            utils.mkdir(save_path)
            hist_kwargs = dict(bins='auto', density=True)
            label = 'noise-std=1_'
            result = self.denoiser_criteria.compute(save_path, label=label, **hist_kwargs)
            self.log(label + "local remainder noise worst p-value", result['remainder_noise_worst_p'],
                     prog_bar=True, logger=True)
            self.log(label + "local remainder noise random p-value", result['remainder_noise_random_p'],
                     prog_bar=True, logger=True)
            self.log(label + "remainder noise overall p-value", result['remainder_noise_overall_p'],
                     prog_bar=True, logger=True)
