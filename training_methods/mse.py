import torch.nn.functional as F
import pytorch_lightning as pl
from nets.factory import factory as nets_fac
from optimization.optimizers import factory as opt_fac
import os
import torch
from pytorch_lightning.metrics import PSNR
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import utils.utils as utils
import math

class MSE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser = nets_fac[config['denoiser_cfg']['type']](config['denoiser_cfg'])
        self.val_collage = None
        self.collages = None
        self.denoised_images_collection = None
        self.psnr = None
        if 'test_cfg' in config:
            self.test_cfg = config['test_cfg']
        self.fid = None
        self.denoiser_criteria = None
        self.test_path = None

    def configure_optimizers(self):
        return opt_fac[self.config['optim_cfg']['type']](self.denoiser.parameters(), self.config['optim_cfg'])

    def forward(self, y, **kwargs):
        out = self.denoiser(y, **kwargs)
        if type(out) is list:
            out = out[0]
        return out

    @staticmethod
    def batch_postprocess(batch):
        return batch['real'], batch['noisy']

    def training_step(self, batch, batch_idx):
        x, y = self.batch_postprocess(batch)
        loss = F.mse_loss(self(y), x)
        self.log('mse_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_collage is None:
            self.val_collage = utils.CollageVal().to(self.device)
        x, y = self.batch_postprocess(batch)
        with torch.no_grad():
            out = self(y)
        self.val_collage.update(x)
        self.val_collage.update(out)

    def validation_epoch_end(self, outputs):
        out = self.val_collage.compute()
        fig = plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title("Denoised Images")
        plt.imshow(np.transpose(vutils.make_grid(out.clamp_(0, 1).detach().cpu(), padding=2,
                                                 normalize=False, range=(0, 1)), (1, 2, 0)))
        fig.savefig(os.path.join(self.val_path, str(self.current_epoch).zfill(5) + "_denoised_collage_" +
                                 datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p") + ".png"), dpi=350)

    def load_from_native_torch(self, path):
        self.denoiser.load_state_dict(torch.load(path, map_location=self.device))

    def on_test_epoch_start(self):
        if self.denoiser_criteria is None:
            avg_kernel = 1 / (3 * 15 * 15) * torch.ones(1, 3, 15, 15).to(self.device)
            self.denoiser_criteria = utils.DenoiserCriteria(avg_kernel).to(self.device)
        if self.test_cfg['collages']:
            self.collages = torch.nn.ModuleDict(
                {str(i): utils.Collage(i, self.test_path, 8, ["real", "noisy", "denoised"]).to(self.device)
                 for i in self.test_cfg['save_batch']})
        if self.test_cfg['fid_and_psnr']:
            self.psnr = PSNR(1).to(self.device)
            self.m_real, self.s_real, model = utils.init_fid(self.test_cfg['training_data_stats_path'],
                                                             self.train_dataloader(),
                                                             self.device,
                                                             verbose=False)
            self.fid = utils.FID(1, self.m_real, self.s_real, model).to(self.device)

    def test_step(self, batch, batch_idx):
        x, y = self.batch_postprocess(batch)
        with torch.no_grad():
            out = self(y)
            if self.test_cfg['fid_and_psnr']:
                self.psnr.update(out, x)
                self.fid.update(out.unsqueeze(0))
            if self.test_cfg['denoiser_criteria']:
                self.denoiser_criteria.update(out - x, y - out, y - x, self.device)
            if self.test_cfg['collages'] and batch_idx in self.test_cfg['save_batch']:
                idx = str(batch_idx)
                self.collages[idx].set_batch_size(x.shape[0])
                self.collages[idx].update("real", x)
                self.collages[idx].update("noisy", y)
                self.collages[idx].update("denoised", out)

    def test_epoch_end(self, outputs):
        if self.test_cfg['fid_and_psnr']:
            self.log("PSNR", self.psnr.compute().item(), prog_bar=True, logger=True)
            self.log("FID", self.fid.compute()[0], prog_bar=True, logger=True)
        if self.test_cfg['collages']:
            for idx in self.collages:
                zfill = max(self.test_cfg['save_batch'])
                self.collages[idx].compute(math.ceil(math.log10(zfill)))
        if self.test_cfg['denoiser_criteria']:
            save_path = os.path.join(self.test_path, "histograms")
            utils.mkdir(save_path)
            hist_kwargs = dict(bins='auto', density=True)
            result = self.denoiser_criteria.compute(save_path, label='OursMSE_', **hist_kwargs)
            self.log("Local remainder noise worst p-value", result['remainder_noise_worst_p'], prog_bar=True,
                     logger=True)
            self.log("Local remainder noise random p-value", result['remainder_noise_random_p'], prog_bar=True,
                     logger=True)
            self.log("Remainder noise overall p-value", result['remainder_noise_overall_p'], prog_bar=True, logger=True)
