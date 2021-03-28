import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class Food11Dataset(Dataset):
    """Dataset for loading food-11 data"""
    def __init__(self, img_dir, label=False, transform=None):
        self.img_dir = img_dir
        self.label = label
        self.transform = transform

        self.img_fnames = [fname for fname in sorted(os.listdir(self.img_dir))
                           if fname.endswith(".jpg")]

    def __getitem__(self, idx):
        img_fname = self.img_fnames[idx]

        # Read image
        img = Image.open(os.path.join(self.img_dir, img_fname))
        # img = img.resize((380, 380), Image.ANTIALIAS)

        if self.transform is not None:
            x = self.transform(img)

        if self.label:
            y = int(img_fname.split('_')[0])
            y = torch.tensor(y, dtype=torch.long)
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.img_fnames)


def get_train_dataloader(img_dir,
                         image_size=(380, 380),
                         batch_size=32,
                         num_workers=16,
                         augment=True,
                         normalize=True,
                         shuffle=True):
    """Create the torch data loader for training set"""
    # Check data augmentation
    if augment:
        augment_transform = transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=image_size,
                                         scale=(0.8, 1.2),
                                         ratio=(0.8, 1.2),
                                         interpolation=Image.ANTIALIAS),
            transforms.RandomAffine(30,
                                    translate=(0.2, 0.2),
                                    scale=(0.9, 1.15),
                                    shear=15)
        ])
        train_transform = transforms.Compose([
            augment_transform,
            transforms.ToTensor(),
        ])
        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(size=image_size,
        #                                  scale=(0.80, 1.20),
        #                                  ratio=(0.80, 1.20),
        #                                  interpolation=Image.ANTIALIAS),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(25),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(size=image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
        ])

    if normalize:
        normalization = transforms.Normalize(
            mean=[0.55474155, 0.45078358, 0.34352523],
            std=[0.2719837, 0.27492649, 0.28205909]
            )
        train_transform = transforms.Compose([train_transform, normalization])

    train_dataset = Food11Dataset(img_dir,
                                  label=True,
                                  transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle
        )
    return train_dataloader


def get_test_dataloader(img_dir,
                        image_size=(380, 380),
                        batch_size=128,
                        num_workers=16,
                        normalize=True):
    """Create the torch data loader for testing set"""
    test_transform = transforms.Compose([
        transforms.Resize(size=image_size, interpolation=Image.ANTIALIAS),
        transforms.ToTensor(),
    ])

    if normalize:
        normalization = transforms.Normalize(
            mean=[0.55474155, 0.45078358, 0.34352523],
            std=[0.2719837, 0.27492649, 0.28205909]
            )
        test_transform = transforms.Compose([test_transform, normalization])

    test_dataset = Food11Dataset(img_dir,
                                 label=False,
                                 transform=test_transform)
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False)
    return test_dataloader
