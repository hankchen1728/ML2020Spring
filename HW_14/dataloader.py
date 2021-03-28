import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
from torchvision import datasets, transforms


class Convert2RGB(object):

    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __call__(self, img):
        # If the channel of img is not equal to desired size,
        # then expand the channel of img to desired size.
        img_channel = img.size()[0]
        img = torch.cat([img] * (self.num_channel - img_channel + 1), 0)
        return img


class Pad(object):

    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # If the H and W of img is not equal to desired size,
        # then pad the channel of img to desired size.
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)


def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        Pad(32),
        Convert2RGB(3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform


class ImageDataset(object):

    def __init__(self, path):

        transform = get_transform()

        self.MNIST_dataset = datasets.MNIST(root=os.path.join(path, "MNIST"),
                                            transform=transform,
                                            train=True,
                                            download=True)

        self.SVHN_dataset = datasets.SVHN(root=os.path.join(path, "SVHN"),
                                          transform=transform,
                                          split='train',
                                          download=True)

        self.USPS_dataset = datasets.USPS(root=os.path.join(path, "USPS"),
                                          transform=transform,
                                          train=True,
                                          download=True)

    def get_datasets(self):
        a = [
            (self.SVHN_dataset, "SVHN"),
            (self.MNIST_dataset, "MNIST"),
            (self.USPS_dataset, "USPS")
        ]
        return a


class Dataloader(object):

    def __init__(self, dataset, batch_size, split_ratio=0.1):
        self.dataset = dataset[0]
        self.name = dataset[1]
        train_sampler, val_sampler = self.split_dataset(split_ratio)

        self.train_dataset_size = len(train_sampler)
        self.val_dataset_size = len(val_sampler)

        self.train_loader = data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=train_sampler)
        self.val_loader = data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=val_sampler)
        self.train_iter = self.infinite_iter()

    def split_dataset(self, split_ratio):
        data_size = len(self.dataset)
        split = int(data_size * split_ratio)
        indices = list(range(data_size))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = sampler.SubsetRandomSampler(train_idx)
        val_sampler = sampler.SubsetRandomSampler(valid_idx)
        return train_sampler, val_sampler

    def infinite_iter(self):
        it = iter(self.train_loader)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(self.train_loader)
