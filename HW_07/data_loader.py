import os
from PIL import Image
from multiprocessing import Pool

import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class DataProcessor(object):
    def __init__(self, img_dir, label=False):
        self.img_dir = img_dir
        self.has_label = label

    def __call__(self, img_fname):
        image = Image.open(os.path.join(self.img_dir, img_fname))
        image_fp = image.fp
        image.load()
        image_fp.close()

        if self.has_label:
            try:
                y = int(img_fname.split('_')[0])
            except Exception:
                y = 0
            return {"img": image, "label": y}

        return {"img": image}


# class Food11Dataset(Dataset):
#     """Dataset for loading food-11 data"""
#     def __init__(self, img_dir, has_label=False, transform=None):
#         self.img_dir = img_dir
#         self.has_label = has_label
#         self.labels = []
#         self.images = []
#         self.transform = transform

#         self.img_fnames = [fname for fname in sorted(os.listdir(self.img_dir))
#                            if fname.endswith(".jpg")]

#         imgProcessor = DataProcessor(img_dir, label=has_label)

#         # Data Reading Parallel
#         with Pool(16) as pool:
#             for data in tqdm.tqdm(pool.imap(imgProcessor, self.img_fnames),
#                                   total=len(self.img_fnames)):
#                 self.images.append(data["img"])
#                 if has_label:
#                     self.labels.append(data["label"])
#             pool.close()

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image = self.images[idx]
#         if self.transform:
#             image = self.transform(image)

#         if self.has_label:
#             return image, torch.tensor(self.labels[idx], dtype=torch.long)
#         return image

#     def __len__(self):
#         return len(self.images)
#

class Food11Dataset(Dataset):
    """Dataset for loading food-11 data"""
    def __init__(self, img_dir, has_label=False, transform=None):
        self.img_dir = img_dir
        self.label = has_label
        self.transform = transform

        self.img_fnames = [fname for fname in sorted(os.listdir(self.img_dir))
                           if fname.endswith(".jpg")]

    def __getitem__(self, idx):
        img_fname = self.img_fnames[idx]

        # Read image
        img = Image.open(os.path.join(self.img_dir, img_fname))

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


def read_all_images(img_dir, has_label=False):
    """Data Reading Parallel"""
    img_fnames = [fname for fname in sorted(os.listdir(img_dir))
                  if fname.endswith(".jpg")]

    imgProcessor = DataProcessor(img_dir, label=has_label)
    images = []
    labels = []

    # Data Reading Parallel
    with Pool(32) as pool:
        for data in tqdm.tqdm(pool.imap(imgProcessor, img_fnames),
                              total=len(img_fnames)):
            images.append(data["img"])
            if has_label:
                labels.append(data["label"])
        pool.close()
    if has_label:
        return images, labels
    return images


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
        augment_transform = transforms.Compose([
            transforms.RandomCrop(
                image_size,
                pad_if_needed=True,
                padding_mode="symmetric"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])
        train_transform = transforms.Compose([
            augment_transform,
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    if normalize:
        normalization = transforms.Normalize(
            mean=[0.55474155, 0.45078358, 0.34352523],
            std=[0.2719837, 0.27492649, 0.28205909]
            )
        train_transform = transforms.Compose([train_transform, normalization])

    train_dataset = Food11Dataset(img_dir=img_dir,
                                  has_label=True,
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
    # test_transform = transforms.Compose([
    #     transforms.Resize(size=image_size, interpolation=Image.ANTIALIAS),
    #     transforms.ToTensor(),
    # ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    if normalize:
        normalization = transforms.Normalize(
            mean=[0.55474155, 0.45078358, 0.34352523],
            std=[0.2719837, 0.27492649, 0.28205909]
            )
        test_transform = transforms.Compose([test_transform, normalization])

    test_dataset = Food11Dataset(img_dir=img_dir,
                                 has_label=False,
                                 transform=test_transform)
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False)
    return test_dataloader
