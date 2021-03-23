import math
import sys
from enum import Enum
import argparse
import json
import torch
import os


class PossibleDatasets(Enum):
    ffhq = 'ffhq'
    bedroom = 'bedroom'
    church = 'church'

    def __str__(self):
        return self.value


class PossibleMethods(Enum):
    pscgan = 'pscgan'
    ours_mse = 'ours_mse'
    lag = 'lag'
    dncnn = 'dncnn'

    def __str__(self):
        return self.value


def add_train_args(arg_parser) -> argparse.ArgumentParser:
    arg_parser.add_argument("--checkpoint", type=str,
                            help="Path to a pretrained checkpoint. Training continues from this point.")
    arg_parser.add_argument("--learning_rate", "-lr", help="Learning rate (default: 1e-3).", type=float, default=1e-3)
    arg_parser.add_argument("--batch_size", "-B", help="(default: 32)", type=int, default=32)
    arg_parser.add_argument("--penalty_batch_size", "-PB",
                            help="Batch size for computing our penalty term (default: 8). Supported only for PSCGAN.",
                            type=int, default=8)
    arg_parser.add_argument("--expansion", "-M", type=int, default=8,
                            help="Number of denoised samples to draw for each noisy image to estimate the mean of the "
                                 "posterior distribution (default: 8). Supported only for PSCGAN.")
    return arg_parser


def add_test_args(arg_parser) -> argparse.ArgumentParser:
    arg_parser.add_argument("--checkpoint", help="Path to a pretrained checkpoint.",
                            type=str, required='--config' not in sys.argv)
    arg_parser.add_argument("--save_batch", type=int, nargs='+',
                            help="Indices of denoised batches to save. -1 to save all of the test set "
                                 "(default: don't save anything).")
    arg_parser.add_argument("--fid_and_psnr", action='store_true',
                            help="Set to measure FID and PSNR.",)
    arg_parser.add_argument("--num_fid_evals", type=int, default=1,
                            help="The number of FID evaluations for PSCGAN/LAG. Used only when --fid_and_psnr is set."
                                 " (default: 1).", )
    arg_parser.add_argument("--sigma_z", type=float, default=0, nargs='+',
                            help="A list of noise injection std's for testing the PSNR and FID of PSCGAN/LAG. "
                                 "Used when --fid_and_psnr is set. Also used by --save_batch (default: [1]).")
    arg_parser.add_argument("--num_avg_samples_traversal", "-N", type=int, default=0, nargs='+',
                            help="List of expansions on which to test PSCGAN-A/LAG-A. "
                                 "Used when --fid_and_psnr is set (default: [64]).")

    arg_parser.add_argument("--denoiser_criteria", help="Set to run the denoising criteria evaluation.", action='store_true')
    arg_parser.add_argument("--divide_expanded_forward_pass", type=int, default=1,
                            help="In cases where many expansions are made, the GPU requirements might be too large for"
                                 "a single forward pass. You can break a forward pass into multiple passes with"
                                 "this argument. Make sure that it is either 1 or divisible by 2 (default: 1).")
    arg_parser.add_argument("--test_set", help="Test set folder path.", type=str)

    return arg_parser


def add_base_args(parent, **kwargs) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(parents=[parent])
    arg_parser.add_argument("--method", help="The denoising algorithm to use.",
                            type=PossibleMethods, choices=list(PossibleMethods), required='--config' not in sys.argv)
    arg_parser.add_argument("--noise_std", help="The standard deviation of the additive white Gaussian noise "
                                                "contamination.",
                            type=int, required='--config' not in sys.argv)
    return arg_parser


def add_optional_args(arg_parser):
    arg_parser.add_argument("--train_set", help="Training set folder path.", type=str)
    arg_parser.add_argument("--config", help="Path to the configuration file of this execution.", type=str)
    arg_parser.add_argument("--n_gpus", help="number of GPUs to use (default: 1).", type=int, default=1)
    arg_parser.add_argument("--out_dir", help="Output directory of the results (default is the parent directory of the "
                                              "config, if given, else it will override experiements/results).", type=str)
    return arg_parser


def handle_arg_errors(arg_parser, config, params, test_flag):
    error_msg = ""
    if test_flag and config['training_checkpoint_path'] is None:
        error_msg = "No checkpoint given to test.py. Provide a .ckpt path with the --ckpt flag."

    if config['dataset_cfg']['training_set_path'] is None:
        if not test_flag:
            error_msg = "No training set provided. Provide a training set path with the --train_set flag."
        elif config['fid_and_psnr']:
            error_msg = "A training set is needed to compute the FID when --fid_and_psnr is set. " \
                        "Provide a training set path with the --train_set flag."

    if config['dataset_cfg']['test_set_path'] is None and test_flag:
        if test_flag:
            error_msg = "No test set provided. Provide a test set path with the --test_set flag."

    if test_flag:
        test_cfg = config['training_method_cfg']['test_cfg']
        if not any([test_cfg['collages'], test_cfg['fid_and_psnr'],
                                  test_cfg['denoiser_criteria']]):
            error_msg = "No test option was given. Set at least one test option:" \
                        "\n{--save_batch, --fid_and_psnr, --denoiser_criteria}."
        if test_cfg['collages']:
            if -1 in test_cfg["save_batch"] and len(test_cfg["save_batch"]) > 1:
                error_msg = "When providing -1 to --save_batch, please do not provide additional values. " \
                            "I can't tell which option you want!"
            for elem in test_cfg["save_batch"]:
                if elem < -1:
                    error_msg = "Unrecognized value provided to --save_batch"

        if test_cfg['fid_and_psnr'] and test_cfg['num_fid_evals'] > 1 and config["training_method"] != "gan":
            error_msg = "Average FID over N > 1 possible outcomes is not supported for MSE based methods. Please set " \
                        "--num_fid_evals 1 when --fid_and_psnr is set."

    else:
        if params.method != PossibleMethods.pscgan and ('--expansion' in sys.argv or '-M' in sys.argv):
            error_msg = "Used --expansion (-M) flag for a method other than PSCGAN. Unset this flag to continue."

        if params.method != PossibleMethods.pscgan and ('--penalty_batch_size' in sys.argv or '-PB'
                                                                          in sys.argv):
            error_msg = "Used --penalty_batch_size (-PB) flag for a method other than PSCGAN. Unset this flag to continue."

    if params.n_gpus < 1:
        error_msg = "You must have at least one GPU to perform training/testing."

    if params.n_gpus > torch.cuda.device_count():
        error_msg = f"You have requested more GPUs than detected by torch." \
                    f"\nRequested: {params.n_gpus}, Detected: {torch.cuda.device_count()}"

    if params.method == PossibleMethods.pscgan:
        pb = config['training_method_cfg']['gen_cfg']['loss_cfg']['penalty_batch_size']
        if pb % params.n_gpus > 0:
            error_msg = f"--penalty_batch_size (-PB) must be divisible by the number of GPUs.\nPB: {pb}, " \
                        f"number of gpus: {params.n_gpus}\n To continue, run with --penalty_batch_size PB " \
                        f"--n_gpus N and make sure that PB % N == 0."

    if config['dataset_cfg']['train_batch_size'] % params.n_gpus > 0:
        error_msg = f"--batch_size (-B) must be divisible by the number of GPUs.\nB: " \
                    f"{config['dataset_cfg']['train_batch_size']}, number of gpus: {params.n_gpus}\n To continue," \
                    f" run with --batch_size B --n_gpus N and make sure that B % N == 0."

    if error_msg != "":
        arg_parser.error(error_msg+"\nAborting")
        sys.exit(1)


def override_config(arg_parser, params, test_flag):
    if '--config' not in sys.argv:
        if params.method == PossibleMethods.ours_mse or params.method == PossibleMethods.dncnn:
            params.config = './experiments/default_mse_config.json'
        else:
            params.config = './experiments/default_config.json'

    with open(params.config) as f:
        config = json.load(f)

    if params.method == PossibleMethods.dncnn:
        config['training_method_cfg']['denoiser_cfg']['type'] = "dncnn"
    if params.method == PossibleMethods.lag:
        config['training_method_cfg']['gen_cfg']['loss_cfg']['type'] = "gen_lag_loss"

    output_dir = os.path.dirname(params.config)
    if '--noise_std' in sys.argv:
        config['dataset_cfg']['noise_std_dev'] = params.noise_std
    if '--checkpoint' in sys.argv:
        config['training_checkpoint_path'] = params.checkpoint
    if '--train_set' in sys.argv:
        config['dataset_cfg']['training_set_path'] = params.train_set
    if '--test_set' in sys.argv:
        config['dataset_cfg']['test_set_path'] = params.test_set
        if not ('--config' not in sys.argv and (not test_flag or '--fid_and_psnr' in sys.argv)):
            config['dataset_cfg']['training_set_path'] = params.test_set
    if "--sigma_z" not in sys.argv:
        params.sigma_z = [1]
    if "--num_avg_samples_traversal" not in sys.argv and "-N" not in sys.argv:
        params.num_avg_samples_traversal = [64]

    if test_flag:
        config['training_method_cfg']['test_cfg'] = {
            "collages": params.save_batch is not None,
            "save_batch": set(params.save_batch) if params.save_batch is not None else set(),
            "fid_and_psnr": params.fid_and_psnr,
            "denoiser_criteria": params.denoiser_criteria,
            "num_fid_evals": params.num_fid_evals,
            "noise_std_traversal": params.sigma_z,
            "num_avg_samples_traversal": params.num_avg_samples_traversal,
            "divide_expanded_forward_pass": params.divide_expanded_forward_pass,
            "training_data_stats_path": os.path.join(str(config['dataset_cfg']['training_set_path']) + '_stats')
        }
    else:
        if '--learning_rate' in sys.argv or '-lr' in sys.argv:
            config['training_method_cfg']['optim_cfg']['lr'] = params.learning_rate
        if '--batch_size' in sys.argv or '-B' in sys.argv:
            config['dataset_cfg']['train_batch_size'] = params.batch_size
        if params.method == PossibleMethods.pscgan and ('--expansion' in sys.argv or '-M' in sys.argv):
            config['training_method_cfg']['gen_cfg']['loss_cfg']['expansion'] = params.expansion
        if params.method == PossibleMethods.pscgan and ('--penalty_batch_size' in sys.argv or '-PB' in sys.argv):
            config['training_method_cfg']['gen_cfg']['loss_cfg']['penalty_batch_size'] = params.penalty_batch_size

    handle_arg_errors(arg_parser, config, params, test_flag)
    return config, output_dir
