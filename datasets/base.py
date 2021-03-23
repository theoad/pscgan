import torch.utils.data as data
import torch
from utils import utils


class DatasetBase(data.Dataset):
    def __init__(self, config):
        self.dataset_path = config['path']
        self.num_image_channels = config['num_image_channels']
        self.noise_std_dev = config['noise_std_dev']
        self.images_paths = utils.get_images_paths(self.dataset_path)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        image = utils.imread_uint(self.images_paths[index], self.num_image_channels)
        # TODO: add augmentation
        # mode = np.random.randint(0, 8)
        # image = utils.augment_img(image, mode=mode)
        image = utils.uint2tensor3(image)

        # TODO: should we normalize back to [0,1]?
        noisy_image = image + torch.randn(image.shape) * (self.noise_std_dev / 255.0)

        return {'real': image, 'noisy': noisy_image}
