from pytorch_lightning import Callback
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import utils
import os
from datetime import datetime
from datasets.factory import factory as ds_fac
from torch.optim.lr_scheduler import MultiStepLR
import warnings
warnings.filterwarnings("ignore")


class ChangeOptimizerLR(Callback):
    def __init__(self, config):
        self.lr = config['scaled_lr']

    def on_train_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr


factory = {
    'change_lr': ChangeOptimizerLR
}
