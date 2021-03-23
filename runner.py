import os
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import utils.utils as utils
from datasets.factory import factory as ds_fac
from training_methods.factory import factory as tm_fac
from utils.callbacks import factory as cb_fac
from datetime import datetime
from pytorch_lightning.plugins import DDPPlugin
import utils.argparse_utils as argparse_utils
from utils.argparse_utils import PossibleMethods


def runner(parent, params, test_flag=False) -> None:
    config, experiment_dir = argparse_utils.override_config(parent, params, test_flag)
    device_ids = list(range(params.n_gpus))
    num_gpus = len(device_ids)
    accelerator = config['accelerator']
    if accelerator != 'ddp':
        device_ids = device_ids[0]
    seed_everything(0)  # Required for Distributed Data Parallel

    datamodule = ds_fac[config['dataset_cfg']['type']](config['dataset_cfg'], num_gpus)
    datamodule.setup()

    # Create logger
    if params.out_dir is None:
        results_path = os.path.join(experiment_dir, 'results')
    else:
        results_path = params.out_dir
    utils.mkdir(results_path)

    callbacks = []
    if test_flag:
        num_batches = datamodule.test_dataloader().__len__()

        save_batch = config['training_method_cfg']['test_cfg']['save_batch']
        if -1 in save_batch:
            save_batch = {range(num_batches)}
        save_batch_cpy = save_batch
        for batch_idx in save_batch_cpy:
            if batch_idx >= num_batches:
                save_batch.remove(batch_idx)
        config['training_method_cfg']['test_cfg']['save_batch'] = save_batch

    # scaling of the learning rate and the penalty batch size according to the number of GPUs
    config['scaled_lr'] = config['training_method_cfg']['optim_cfg']['lr'] * num_gpus

    if params.method == PossibleMethods.pscgan:
        config['training_method_cfg']['gen_cfg']['loss_cfg']['penalty_batch_size'] = \
            config['training_method_cfg']['gen_cfg']['loss_cfg']['penalty_batch_size'] // num_gpus
    for callback in config['callbacks']:
        callbacks.append(cb_fac[callback](config))

    denoiser = tm_fac[config['training_method']](config['training_method_cfg'])
    if config['training_checkpoint_path'] is not None:
        denoiser = tm_fac[config['training_method']].load_from_checkpoint(config['training_checkpoint_path'],
                                                                          strict=False,
                                                                          config=config['training_method_cfg'])
    trainer = pl.Trainer(gpus=device_ids,
                         plugins=DDPPlugin(find_unused_parameters=True) if accelerator == 'ddp' else None,
                         max_epochs=config['num_train_epochs'],
                         accelerator=accelerator,
                         log_every_n_steps=config['log_every'],
                         callbacks=callbacks,
                         default_root_dir=results_path,
                         limit_val_batches=1,
                         check_val_every_n_epoch=5,
                         resume_from_checkpoint=config['training_checkpoint_path'])

    denoiser.val_path = results_path
    denoiser.test_path = os.path.join(results_path, datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))
    if test_flag:
        trainer.test(denoiser, datamodule=datamodule)
    else:
        trainer.fit(denoiser, datamodule=datamodule)
