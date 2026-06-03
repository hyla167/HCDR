"""Datasets, per-dataset configuration, transforms and the weighted subset.

All custom datasets expose a ``.targets`` attribute (list / ndarray of integer
labels) so the rest of the pipeline can treat them uniformly with the
torchvision CIFAR datasets.
"""
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets as tvdatasets
from PIL import Image


# --------------------------------------------------------------------------- #
# Per-dataset configuration                                                    #
# --------------------------------------------------------------------------- #
# n_cls, cls_per_task, image size, default task-0 epochs / per-task epochs /
# batch size.  epochs / start_epoch / batch_size are *defaults*: a value passed
# on the command line overrides them (see config.py).
DATASET_CONFIG = {
    'cifar10':       dict(n_cls=10,  cls_per_task=2,  size=64, start_epoch=500, epochs=100, batch_size=512),
    'cifar100':      dict(n_cls=100, cls_per_task=20, size=64, start_epoch=500, epochs=100, batch_size=512),
    'tiny-imagenet': dict(n_cls=200, cls_per_task=20, size=64, start_epoch=500, epochs=50,  batch_size=1024),
    'stl10':         dict(n_cls=10,  cls_per_task=2,  size=64, start_epoch=200, epochs=100, batch_size=128),
    'caltech256':    dict(n_cls=256, cls_per_task=16, size=64, start_epoch=500, epochs=100, batch_size=512),
}

# (mean, std) per dataset
DATASET_STATS = {
    'cifar10':       ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100':      ((0.5153, 0.4961, 0.4497), (0.2608, 0.2551, 0.2779)),
    'tiny-imagenet': ((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)),
    'stl10':         ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    'caltech256':    ((0.5520, 0.5330, 0.5050), (0.3090, 0.3040, 0.3180)),
}

# Google-Drive file id of the processed Tiny-ImageNet archive
_TINY_IMAGENET_FILE_ID = '1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj'


# --------------------------------------------------------------------------- #
# Custom datasets                                                              #
# --------------------------------------------------------------------------- #
class TinyImagenet(Dataset):
    """Tiny-ImageNet, packaged as a torchvision-style dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from googledrivedownloader import download_file_from_google_drive
                print('Downloading dataset')
                download_file_from_google_drive(
                    file_id=_TINY_IMAGENET_FILE_ID,
                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        split = 'train' if train else 'val'
        self.data = np.concatenate([
            np.load(os.path.join(root, 'processed/x_%s_%02d.npy' % (split, n + 1)))
            for n in range(20)])
        self.targets = np.concatenate([
            np.load(os.path.join(root, 'processed/y_%s_%02d.npy' % (split, n + 1)))
            for n in range(20)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(255 * self.data[index]))
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if hasattr(self, 'logits'):
            return img, target, img.copy(), self.logits[index]
        return img, target


class STL10Wrapper(Dataset):
    """Thin wrapper exposing STL10 with a ``.targets`` list."""

    def __init__(self, root, train=True, transform=None, download=True):
        split = 'train' if train else 'test'
        self.dataset = tvdatasets.STL10(root=root, split=split,
                                        transform=transform, download=download)
        self.targets = self.dataset.labels.tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class Caltech256Wrapper(Dataset):
    """Caltech-256 with a reproducible per-class 80/20 split and ``.targets``.

    The clutter category (label 256) is dropped, leaving 256 classes (0-255).
    Train and test instances with the same root/seed produce complementary,
    non-overlapping index sets so replay indices stay consistent across calls.
    """

    def __init__(self, root, train=True, transform=None, download=False,
                 train_ratio=0.8, seed=42):
        base = tvdatasets.Caltech256(root=root, download=download)
        all_y = np.array(base.y)
        valid_idx = np.where(all_y < 256)[0]
        valid_y = all_y[valid_idx]

        rng = np.random.RandomState(seed)
        split_idx = []
        for cls in range(256):
            cls_idx = valid_idx[valid_y == cls]
            cls_idx = cls_idx[rng.permutation(len(cls_idx))]
            cut = max(1, int(len(cls_idx) * train_ratio))
            split_idx.extend((cls_idx[:cut] if train else cls_idx[cut:]).tolist())

        self._base = base
        self._split_idx = split_idx
        self.targets = all_y[split_idx].tolist()
        self.transform = transform

    def __len__(self):
        return len(self._split_idx)

    def __getitem__(self, idx):
        img, _ = self._base[self._split_idx[idx]]
        target = self.targets[idx]
        img = img.convert('RGB')                  # some Caltech images are grayscale
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class WeightedSubset(Dataset):
    """A subset that also returns a per-sample importance weight and the original
    dataset index: ``(image, label, weight, original_index)``."""

    def __init__(self, dataset, indices, weights):
        self.dataset = dataset
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]
        return image, label, self.weights[idx], original_idx

    def __len__(self):
        return len(self.indices)


# --------------------------------------------------------------------------- #
# Factories                                                                    #
# --------------------------------------------------------------------------- #
def build_dataset(opt, train, transform):
    """Instantiate the raw dataset for ``opt.dataset`` (no subsetting)."""
    d, root = opt.dataset, opt.data_folder
    if d == 'cifar10':
        return tvdatasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    if d == 'cifar100':
        return tvdatasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    if d == 'tiny-imagenet':
        return TinyImagenet(root=root, train=train, transform=transform, download=True)
    if d == 'stl10':
        return STL10Wrapper(root=root, train=train, transform=transform, download=True)
    if d == 'caltech256':
        return Caltech256Wrapper(root=root, train=train, transform=transform, download=False)
    raise ValueError('dataset not supported: {}'.format(d))


def get_targets(dataset):
    return np.asarray(dataset.targets)


def get_normalize(opt):
    if opt.dataset in DATASET_STATS:
        mean, std = DATASET_STATS[opt.dataset]
    elif opt.dataset == 'path':
        mean, std = eval(opt.mean), eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    return transforms.Normalize(mean=mean, std=std)


def build_train_transform(opt):
    blur_p = 0.5 if opt.size > 32 else 0.0
    return transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size,
                                     scale=(0.1 if opt.dataset == 'tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=opt.size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=blur_p),
        transforms.ToTensor(),
        get_normalize(opt),
    ])


def build_eval_transform(opt):
    return transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        get_normalize(opt),
    ])