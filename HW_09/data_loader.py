import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def image_process(img_array: np.ndarray):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      img_array: Array of images (N, H, W, C)
    Returns:
      img_array: Array of images (N, C, H, W)
    """
    img_array = img_array.copy()
    img_array = img_array.transpose((0, 3, 1, 2))
    img_array = (img_array / 255.0) * 2 - 1
    img_array = img_array.astype(np.float32)
    return img_array


class ImageDataset(Dataset):
    """A dataset generating processed images"""
    def __init__(self, img_array: np.ndarray):
        # TODO: add image augmentation transformation
        self.img_array = img_array
        self.img_size = self.img_array.shape[2:]
        self.num_image = self.img_array.shape[0]

    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):
        return self.img_array[idx]


def get_train_valid_dataLoader(
        npy_fpath,
        batch_size=8,
        train_shuffle=True,
        valid_split=0.1,
        num_workers=16,
        rand_seed=20):
    # Load image array and preprocessing
    img_array = np.load(npy_fpath)
    img_array = image_process(img_array)

    to_split_valid = isinstance(valid_split, float) and (0 < valid_split < 1)

    # Split image array w.r.t. train and valid part
    if to_split_valid:
        num_imgs = img_array.shape[0]
        rand_indices = np.arange(num_imgs)
        np.random.seed(seed=rand_seed)
        np.random.shuffle(rand_indices)
        num_train = int(num_imgs * (1 - valid_split))

        # Split train and valid set
        X_train = img_array[rand_indices[:num_train]]
        X_val = img_array[rand_indices[num_train:]]
    else:
        X_train = img_array

    # Create train dataset and data loader
    train_dataset = ImageDataset(X_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        pin_memory=True,
        num_workers=num_workers)

    if to_split_valid:
        valid_dataset = ImageDataset(X_val)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=256,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers)
        return train_loader, valid_loader

    return train_loader
