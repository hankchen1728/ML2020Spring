import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


def config_source_dataloader(
        img_folder: str,
        batch_size=32,
        valid_size=0.1,
        shuffle=False,
        random_seed=10):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size < 1)), error_msg
    to_split_valid = isinstance(valid_size, float) and (0 < valid_size < 1)

    # Transformations for dataloader
    source_transform = transforms.Compose([
        # To grayscale image
        transforms.Grayscale(),
        transforms.Lambda(
            lambda x: cv2.Canny(cv2.blur(np.array(x), (2, 2)), 170, 300)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    source_dataset = ImageFolder(img_folder, transform=source_transform)

    num_samples = 5000
    indices = np.arange(num_samples)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if to_split_valid:
        split = int(np.floor(valid_size * num_samples))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_dataset = Subset(source_dataset, train_idx)
        valid_transform = transforms.Compose([
            # To grayscale image
            transforms.Grayscale(),
            transforms.Lambda(
                lambda x: cv2.Canny(cv2.blur(np.array(x), (2, 2)), 170, 300)),
            transforms.ToTensor(),
        ])
        valid_dataset = ImageFolder(img_folder, transform=valid_transform)
        valid_dataset = Subset(valid_dataset, valid_idx)
    else:
        train_dataset = source_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True
        )

    if to_split_valid:
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size*4,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True
            )
        return train_dataloader, valid_dataloader

    return train_dataloader


def config_target_dataloader(
        img_folder: str,
        batch_size=32,
        augment=False,
        shuffle=False):
    target_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
    ])

    if augment:
        target_transform = transforms.Compose([
            target_transform,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    else:
        target_transform = transforms.Compose([
            target_transform,
            transforms.ToTensor()
        ])

    target_dataset = ImageFolder(img_folder, transform=target_transform)
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True
        )
    return target_dataloader
