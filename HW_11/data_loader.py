import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    # def BGR2RGB(self, img):
    #     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    dataset = FaceDataset(fnames, transform)
    return dataset
