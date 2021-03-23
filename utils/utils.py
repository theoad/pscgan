import numpy as np
import torch
import cv2
import logging
import os
import math
import shutil
from pytorch_lightning.metrics.metric import Metric
import matplotlib.pyplot as plt
from typing import Any, Optional
import torchvision.utils as vutils
from collections import abc
from utils.fid import calculate_frechet_distance
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.functional import conv2d
from scipy import stats
from nets.inception import InceptionV3
from utils.original_fid.fid_score import compute_statistics_of_path
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


class DenoiserCriteria(Metric):
    def __init__(self,
                 kernel,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,)
        self.add_state("histogram_residual", default=[], dist_reduce_fx=None)
        self.add_state("histogram_remainder_noise", default=[], dist_reduce_fx=None)
        self.add_state("histogram_noise", default=[], dist_reduce_fx=None)
        self.add_state("worst_p_values", default=[], dist_reduce_fx=None)
        self.add_state("random_p_values", default=[], dist_reduce_fx=None)
        self.add_state("overall_p_values", default=[], dist_reduce_fx=None)
        self.avg_kernel = kernel

    def get_p_value_of_patch(self, res_idx, remainder_noise):
        patch_spatial_extent = self.avg_kernel.shape[-1]
        residual_spatial_extent = remainder_noise.shape[-1] - patch_spatial_extent + 1
        row, col = res_idx // residual_spatial_extent, res_idx % residual_spatial_extent
        _, p = stats.normaltest(remainder_noise[:, row:row + patch_spatial_extent,
                                col:col + patch_spatial_extent].flatten().cpu().numpy())
        return p

    def update(self, residual, remainder_noise, noise, device):
        batch_size = residual.shape[0]
        residual_energy = conv2d(residual**2, self.avg_kernel).view(batch_size, -1) ** 0.5
        remainder_noise_energy = conv2d(remainder_noise**2, self.avg_kernel) ** 0.5
        noise_energy = conv2d(noise**2, self.avg_kernel)**0.5
        # histograms
        self.histogram_residual.append(residual_energy.flatten())
        self.histogram_remainder_noise.append(remainder_noise_energy.flatten())
        self.histogram_noise.append(noise_energy.flatten())
        # normality test
        tot_patches = residual_energy.shape[1]
        num_samples = 20
        for i in range(residual_energy.shape[0]):
            _, worst_indices = torch.topk(residual_energy[i], num_samples)
            rand_indices = torch.randint(high=tot_patches, size=(num_samples,))
            self.worst_p_values.append(torch.tensor([self.get_p_value_of_patch(wi, remainder_noise[i])
                                                     for wi in worst_indices]).to(device))
            self.random_p_values.append(torch.tensor([self.get_p_value_of_patch(ri, remainder_noise[i])
                                                      for ri in rand_indices]).to(device))
        self.overall_p_values.append(torch.tensor([stats.normaltest(est_noise.flatten().cpu().numpy())[1]
                                                   for est_noise in remainder_noise]).to(device))

    def compute(self, save_path, label, **hist_kwargs):
        res_data = torch.cat(self.histogram_residual, dim=0).cpu().numpy()
        remainder_noise_data = torch.cat(self.histogram_remainder_noise, dim=0).cpu().numpy()
        noise_data = torch.cat(self.histogram_noise, dim=0).cpu().numpy()
        plt.style.use("ggplot")
        counts, bins = np.histogram(remainder_noise_data, **hist_kwargs)
        _ = plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label='local-remainder-noise-RMS')
        save_hist(os.path.join(save_path, label + 'local-remainder-noise-RMS.txt'), counts, bins)
        
        counts, bins = np.histogram(noise_data, **hist_kwargs)
        _ = plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label='local-noise-RMS')
        save_hist(os.path.join(save_path, label + 'local-noise-RMS.txt'), counts, bins)

        counts, bins = np.histogram(res_data, **hist_kwargs)
        _ = plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label='patch-RMSE')
        save_hist(os.path.join(save_path, label + 'patch-RMSE.txt'), counts, bins)

        plt.legend(loc='upper left')
        plt.savefig(os.path.join(save_path, label + "histograms.png"))
        plt.close()

        worst_p = torch.cat(self.worst_p_values).cpu().numpy()
        random_p = torch.cat(self.random_p_values).cpu().numpy()
        overall_p = torch.cat(self.overall_p_values).cpu().numpy()
        return {
            'remainder_noise_worst_p': np.sum(worst_p > 0.05)/len(worst_p) * 100,
            'remainder_noise_random_p': np.sum(random_p > 0.05)/len(random_p) * 100,
            'remainder_noise_overall_p': np.sum(overall_p > 0.05)/len(overall_p) * 100
        }


class FID(Metric):
    def __init__(self,
                 expansion: int,
                 m_real,
                 s_real,
                 model,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,)
        self.expansion = expansion
        for i in range(expansion):
            self.add_state("fake_pred_arr" + str(i), default=[], dist_reduce_fx=None)
        self.m_real, self.s_real = m_real, s_real
        self.model = model

    def get_pred(self, batch):
        with torch.no_grad():
            pred = self.model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)
        return pred

    def update(self, expanded_reshaped_fake):
        for i in range(self.expansion):
            fake_pred = self.get_pred(expanded_reshaped_fake[i])
            getattr(self, "fake_pred_arr" + str(i)).append(fake_pred)

    def compute(self):
        fid_scores = []
        for i in range(self.expansion):
            act_fake = torch.cat(getattr(self, "fake_pred_arr" + str(i)), dim=0).cpu().numpy()
            mu_fake = np.mean(act_fake, axis=0)
            sigma_fake = np.cov(act_fake, rowvar=False)
            fid = calculate_frechet_distance(self.m_real, self.s_real, mu_fake, sigma_fake)
            fid_scores.append(fid)
        return torch.tensor(fid_scores)


class Collage(Metric):
    def __init__(self,
                 id: int,
                 path: str,
                 expansion: int,
                 attr_names: list,
                 save_batch: bool = True,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,)
        for name in attr_names:
            self.add_state(name, default=[], dist_reduce_fx=None)
        self.id = id
        self.expansion = expansion
        self.batch_size = 0
        self.attr_names = attr_names
        self.path = path
        self.save_batch = save_batch

    def update(self, attr_name, batch):
        getattr(self, attr_name).append(batch)

    def set_batch_size(self, batch_size):
        self.batch_size += batch_size

    def set_id(self, new_id: int):
        self.id = new_id

    def compute(self, zfill):
        if self.save_batch:
            for j in range(len(getattr(self, self.attr_names[0]))):
                for i in range(self.batch_size):
                    path = os.path.join(self.path, str(self.id).zfill(zfill), str(self.batch_size * j + i))
                    mkdir(path, remove=False)
                    for name in self.attr_names:
                        if getattr(self, name)[j].ndim == 5:
                            imgs_to_save = [getattr(self, name)[j][k][i] for k in range(self.expansion)]
                        else:
                            imgs_to_save = [getattr(self, name)[j][i]]
                        save_batch(imgs_to_save, path, label=name, rgb='std' not in name)
        for name in self.attr_names:
            path = os.path.join(self.path, str(self.id).zfill(zfill))
            mkdir(path, remove=False)
            collage = torch.cat(getattr(self, name), dim=0)
            if collage.ndim == 5:
                collage = reshape_4d_batch(collage)
            fig = plt.figure(figsize=(15, 15))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(collage.clamp_(0, 1).detach().cpu(), padding=2,
                                                     normalize=False, range=(0, 1)), (1, 2, 0)))
            fig.savefig(os.path.join(path, f"collage_{name}.png"), dpi=250)


class Collection(Collage):
    def compute(self):
        path = os.path.join(self.path)
        mkdir(path, remove=False)
        for j in range(len(getattr(self, self.attr_names[0]))):
            for i in range(self.batch_size):
                for name in self.attr_names:
                    if getattr(self, name)[j].ndim == 5:
                        imgs_to_save = [getattr(self, name)[j][k][i] for k in range(self.expansion)]
                    else:
                        imgs_to_save = [getattr(self, name)[j][i]]
                    save_batch(imgs_to_save, path, label=name + str(self.batch_size * j + i), rgb=True)


class CollageVal(Metric):
    def __init__(self, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,)
        self.add_state("images", default=[], dist_reduce_fx=None)

    def update(self, batch: torch.Tensor):
        self.images.append(batch)

    def compute(self):
        return torch.cat(self.images, dim=0)


def init_fid(stats_path, train_dataloader, device, verbose=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    if os.path.exists(os.path.join(stats_path, 'm.npy')):
        m_real = np.load(os.path.join(stats_path, 'm.npy'))
        s_real = np.load(os.path.join(stats_path, 's.npy'))
        if verbose:
            print("Loaded real data mu and sigma...", flush=True)
    else:
        mkdir(stats_path, remove=False)
        m_real, s_real = compute_statistics_of_path(train_dataloader, model, 128, 2048, device)
        np.save(os.path.join(stats_path, 'm'), m_real)
        np.save(os.path.join(stats_path, 's'), s_real)
        if verbose:
            print("Calculated real data mu and sigma...", flush=True)
    return m_real, s_real, model


def save_hist(save_path, hist, bin_edges, delimiter=' '):
    with open(save_path, "w") as hist_file:
        to_write = [f'bin{delimiter}edge\n']
        to_write += [f'{float(h)}{delimiter}{float(e)}\n' for h, e in zip(hist, bin_edges)]
        hist_file.writelines(to_write)


def reshape_4d_batch(batch):
    return batch.reshape(batch.shape[0] * batch.shape[1],
                         batch.shape[2],
                         batch.shape[3],
                         batch.shape[4])

def expand_4d_batch(batch, n):
    if n == 0:
        return batch
    return reshape_4d_batch(batch.unsqueeze(0).expand(n, -1, -1, -1, -1))


def expand_4d_batch_cat(batch, num_items_to_expand, n):
    if n == 0:
        return batch
    return torch.cat((expand_4d_batch(batch[:num_items_to_expand], n - 1), batch), dim=0)


def restore_expanded_4d_batch(expanded_batch, n):
    if n == 0:
        return expanded_batch
    return expanded_batch.reshape(n,
                                  expanded_batch.shape[0] // n,
                                  expanded_batch.shape[1],
                                  expanded_batch.shape[2],
                                  expanded_batch.shape[3])


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def imsave(img, img_path, rgb=True):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    if rgb:
        cv2.imwrite(img_path, img)
    else:
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_image = (255 - gray_image)
        cv2.imwrite(img_path, gray_image)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_images_paths(folder_path):
    if folder_path is None:
        return None
    assert os.path.isdir(folder_path), '{:s} is not a valid directory'.format(folder_path)
    images_paths = []
    for dir_path, _, file_names in sorted(os.walk(folder_path)):
        for file_name in sorted(file_names):
            if is_image_file(file_name):
                img_path = os.path.join(dir_path, file_name)
                images_paths.append(img_path)
    assert images_paths, '{:s} does not contain any valid image file'.format(folder_path)
    return sorted(images_paths)


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def logger(logger_name, log_path='info_logger.log', level='info'):
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = levels[level]
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


def mkdir(path, remove=False):
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
        else:
            return

    os.makedirs(path)


def cycle_iterable(iterable):
    while True:
        for x in iterable:
            yield x


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def save_batch(batch, save_path=None, label="", rgb=True):
    im_list = []
    for i, image in enumerate(batch):
        im = tensor2uint(image.to(torch.device("cpu")))
        im_list.append(im)
        if save_path is not None:
            full_label = ""
            if len(batch) > 1:
                full_label += str(i).zfill(2)
            full_label += label + ".png"
            imsave(im, os.path.join(save_path, full_label), rgb)


def nested_dict_iter(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        else:
            yield key, value
