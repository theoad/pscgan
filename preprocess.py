import argparse
import os
import sys
from PIL import Image
from io import BytesIO
import math
from tqdm import tqdm
import lmdb
from multiprocessing.dummy import Pool as ThreadPool
import time
import torch
import json
from pytorch_lightning import seed_everything
from utils.argparse_utils import PossibleDatasets
from utils.utils import mkdir, init_fid
from datasets.factory import factory as ds_fac


def prepare_lsun(args):
    env = lmdb.open(args.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
    for fsize, mode in [(4000, 'test'), (100000, 'train')]:
        print(f'Organising {args.dataset} {mode} set')
        start = time.time()
        out_path = os.path.join(args.out_dir, mode, 'dummy')
        mkdir(out_path, remove=True)
        num_zeros = math.ceil(math.log10(fsize))

        with open(f"utils/{args.dataset}_{mode}_keys.txt", 'r') as fp:
            pbar = tqdm(total=fsize, desc='test')

            def read_lmdb_item(idx, key):
                with env.begin(write=False) as txn:
                    key_b = bytes(key[:-1].encode('utf-8'))
                    val = txn.get(key_b)
                    img = Image.open(BytesIO(val))
                    img.save(os.path.join(out_path, str(idx).zfill(num_zeros) + '.png'), 'PNG')
                    pbar.update(1)

            with ThreadPool(args.n_threads) as pool:
                pool.starmap(read_lmdb_item, enumerate(fp.readlines()))
            pbar.close()
        end = time.time()
        print("elapsed", end - start, "s")
    env.close()


def prepare_ffhq(args):
    train_out_path = os.path.join(args.out_dir, 'train', 'dummy')
    test_out_path = os.path.join(args.out_dir, 'test', 'dummy')
    mkdir(train_out_path, remove=True)
    mkdir(test_out_path, remove=True)
    pbar = tqdm(total=70, desc='splitting FFHQ')
    sys.stdout = open(os.devnull, 'w')

    def copy_ffhq(n, m):
        test = n == 0 and (m == 3 or m == 4)
        print(os.popen(f'cp {args.data_dir}/{n}{m}000/* {test_out_path if test else train_out_path}/').read())
        pbar.update()

    with ThreadPool(args.n_threads) as pool:
        pool.starmap(copy_ffhq, [(n, m) for n in range(7) for m in range(10)])
    pbar.close()


def add_args(parent) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(parents=[parent])
    arg_parser.add_argument("--dataset", help="The dataset to prepare.", required=True,
                            type=PossibleDatasets, choices=list(PossibleDatasets))
    arg_parser.add_argument("--data_dir", type=str, required=True,
                            help="For FFHQ: The directory which contains all 70K images (00000/, 00001/,...,60000/).\n"
                                 "For LSUN: The directory which contains the data.mdb and lock.mdb files")
    arg_parser.add_argument("--out_dir", "-o", default=".", type=str,
                            help="The directory where the script will export the the dataset (default: '.').")
    arg_parser.add_argument("--n_threads", default=32, type=int,
                            help="The number of threads to share lmdb opening workload (default: 32).")
    return arg_parser


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = add_args(parent=parent_parser)

    args = parent_parser.parse_args()

    if args.data_dir == args.out_dir:
        parent_parser.error("OUT_DIR can't be the same as DATA_DIR.\nAborting.")
    for file in os.listdir(args.data_dir):
            if args.dataset == PossibleDatasets.ffhq:
                if not ((os.path.isdir(os.path.join(args.data_dir, file)) and 0 <= int(file) < 70000)
                        or file == "LICENSE.txt"):
                    parent_parser.error("For preprocessing FFHQ, the script expects a folder"
                                        " containing only subdirectories.\nAborting.")
            else:
                if not file.endswith('.mdb'):
                    parent_parser.error("Organizing the LSUN datasets require DATA_DIR to contain .mdb files only.\nAborting.")

    seed_everything(0)
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        parent_parser.error("PyTorch didn't detect any GPU. You must have at least one GPU to prepare the data.\nAborting.")

    if args.dataset == PossibleDatasets.ffhq:
        args.dataset = str(args.dataset)
        prepare_ffhq(args)
    else:
        args.dataset = str(args.dataset)
        prepare_lsun(args)

    with open('experiments/default_config.json') as f:
        default_config = json.load(f)
    ds_config = default_config['dataset_cfg']
    ds_config['train_batch_size'] = 64
    ds_config['training_set_path'] = os.path.join(args.out_dir, 'train')
    ds_config['test_set_path'] = os.path.join(args.out_dir, 'test')
    datamodule = ds_fac[ds_config['type']](ds_config, n_gpus)
    datamodule.setup()
    #TODO: stdout was redirected to devnull, should use stderr instead? fix stdout?
    print('computing train set statistics needed for FID')
    init_fid(os.path.join(str(ds_config['training_set_path']) + '_stats'),
             datamodule.train_dataloader(),
             torch.device('cuda'))
