import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import center_crop
from torchvision.datasets import ImageFolder
from torchvision.datasets import LSUNClass
import torch.utils.data as data
import torch


class CenterCropMinAxis(transforms.CenterCrop):
    def __int__(self):
        super().__init__(-1)

    def __call__(self, img):
        return center_crop(img, (min(img.size), min(img.size)))


class DatasetBase(data.Dataset):
    def __init__(self, dataset_path, target_transform, input_transform, torchvision_class, len_multiplier=1):
        self.dataset_path = dataset_path
        self.dataset = torchvision_class(dataset_path, transform=target_transform)
        self.input_transform = input_transform
        self.len_multiplier = len_multiplier

    def __len__(self):
        return len(self.dataset) * self.len_multiplier

    def __getitem__(self, index):
        target_image, _ = self.dataset[index // self.len_multiplier]
        input_image = self.input_transform(target_image)
        return {'real': target_image,
                'noisy': input_image}


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


torchvision_classes = {
    'image_folder': ImageFolder,
    'lsun_class': LSUNClass
}


class BaseDatamoduleLightning(pl.LightningDataModule):
    def __init__(self, config, num_gpus):
        super().__init__()
        self.training_set_path = config['training_set_path'] if 'training_set_path' in config else None
        self.val_set_path = self.training_set_path
        self.test_set_path = config['test_set_path']
        self.noise_std_dev = config['noise_std_dev']
        self.train_batch_size = config['train_batch_size']
        self.num_workers = config['num_workers']
        self.val_batch_size = config['test_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.accelerator = config['accelerator']
        self.torchvision_class = torchvision_classes[config['torchvision_class']]
        self.len_multiplier = config['len_multiplier'] if 'len_multiplier' in config else 1
        self.num_gpus = num_gpus
        self.training_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    @property
    def train_transforms(self):
        return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    @property
    def val_transforms(self):
        return transforms.Compose([transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([transforms.ToTensor()])

    def setup(self, stage=None):
        noise_transform = AddGaussianNoise(0., self.noise_std_dev / 255.0)
        if self.training_set_path is not None:
            self.training_dataset = DatasetBase(dataset_path=self.training_set_path,
                                                target_transform=self.train_transforms,
                                                input_transform=noise_transform,
                                                torchvision_class=self.torchvision_class,
                                                len_multiplier=self.len_multiplier)
        if self.val_set_path is not None:
            self.val_dataset = DatasetBase(dataset_path=self.val_set_path,
                                           target_transform=self.test_transforms,
                                           input_transform=noise_transform,
                                           torchvision_class=self.torchvision_class,
                                           len_multiplier=1)
        if self.test_set_path is not None:
            self.test_dataset = DatasetBase(dataset_path=self.test_set_path,
                                            target_transform=self.test_transforms,
                                            input_transform=noise_transform,
                                            torchvision_class=self.torchvision_class,
                                            len_multiplier=1)

    def train_dataloader(self):
        if self.training_dataset is not None:
            return DataLoader(self.training_dataset,
                              batch_size=self.train_batch_size // self.num_gpus
                              if self.accelerator == 'ddp' else self.train_batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              drop_last=True,
                              pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size // self.num_gpus
                          if self.accelerator == 'ddp' else self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size // self.num_gpus
                          if self.accelerator == 'ddp' else self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)

    def test_dataloader_custom_batch_size(self, batch_size):
        return DataLoader(self.test_batch_size // self.num_gpus
                          if self.accelerator == 'ddp' else self.test_batch_size,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)


class RandomCropDataset(BaseDatamoduleLightning):
    def __init__(self, config, num_gpus):
        super().__init__(config, num_gpus)

    @property
    def train_transforms(self):
        return transforms.Compose([transforms.RandomCrop(128),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([transforms.RandomCrop(128),
                                   transforms.ToTensor()])


class ResizeDataset(BaseDatamoduleLightning):
    def __init__(self, config, num_gpus):
        super().__init__(config, num_gpus)

    @property
    def train_transforms(self):
        return transforms.Compose([transforms.Resize(128),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([transforms.Resize(128),
                                   transforms.ToTensor()])


class CenterCropResizeDataset(BaseDatamoduleLightning):
    def __init__(self, config, num_gpus):
        super().__init__(config, num_gpus)

    @property
    def train_transforms(self):
        return transforms.Compose([CenterCropMinAxis(-1),
                                   transforms.Resize(128),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([CenterCropMinAxis(-1),
                                   transforms.Resize(128),
                                   transforms.ToTensor()])

    @property
    def val_transforms(self):
        return transforms.Compose([CenterCropMinAxis(-1),
                                   transforms.Resize(128),
                                   transforms.ToTensor()])


class NoTransormDataset(BaseDatamoduleLightning):
    def __init__(self, config, num_gpus):
        super().__init__(config, num_gpus)

    @property
    def train_transforms(self):
        return transforms.Compose([transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([transforms.ToTensor()])