import os
import argparse
from PIL import Image

import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.models as models


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class EvalDataset(Dataset):
    """Dataset for loading evaluation data"""
    def __init__(self, data_dir, out_img_size=(224, 224)):
        # Set file paths
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "images")
        self.label_fpath = os.path.join(self.data_dir, "labels.csv")

        # Read labels
        label_df = pd.read_csv(self.label_fpath)
        labels = label_df.loc[:, "TrueLabel"].to_numpy()
        self.num_imgs = 200
        self.images = np.zeros((self.num_imgs, 224, 224, 3), dtype=np.uint8)
        self.labels = torch.from_numpy(labels).long()

        # Read images
        for i_img in range(self.num_imgs):
            img = Image.open(os.path.join(self.img_dir, "%03d.png" % i_img))
            self.images[i_img] = img

        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )
        self.transform = transforms.Compose([
            # transforms.Resize(out_img_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return self.num_imgs


class FGSM_Attacker:
    def __init__(self, net, img_loader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = net
        self.model.eval()

        self.img_loader = img_loader
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.criterion = nn.CrossEntropyLoss()

    def fgsm_attack(self, image, epsilon, data_grad):
        # Get sign of gradient
        # sign_data_grad = data_grad.sign()
        sign_data_grad = torch.sign(data_grad)
        # Add perturbation to image
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, epsilon, max_iter=1, decode=True):
        assert max_iter >= 1
        adv_images = np.zeros((len(self.img_loader), 224, 224, 3),
                              dtype=np.uint8)

        adv_correct = 0
        org_correct = 0
        total = 0
        for img_idx, (data, target) in tqdm.tqdm(enumerate(self.img_loader)):
            total += 1
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True

            # Get the original predicted class
            output = self.model(data)
            pred = np.argmax(output.data.cpu().numpy())
            if pred == target.data.cpu().numpy()[0]:
                org_correct += 1

            for _ in range(max_iter):
                # Compute gradients
                loss = self.criterion(
                    output,
                    Variable(
                        torch.Tensor([float(target)]).to(self.device).long()
                    )
                )

                self.model.zero_grad()
                loss.backward()
                img_grad = data.grad.data

                # Generate pertured image
                perturbed_data = self.fgsm_attack(data, epsilon, img_grad)

                data.data = perturbed_data.data
                # Predict the pertured image
                output = self.model(perturbed_data)
                pred = np.argmax(output.data.cpu().numpy())

            # Check prediction after attack
            if pred == target.data.cpu().numpy()[0]:
                adv_correct += 1

            adv_img = perturbed_data.data.cpu().numpy()[0]
            adv_img = adv_img.transpose(1, 2, 0)
            if decode:
                adv_img = (adv_img * self.std) + self.mean
                adv_img = adv_img * 255.0
                # adv_img = adv_img[..., ::-1]  # RGB to BGR
                adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)

            adv_images[img_idx] = adv_img

        print("Accuracy before attack: ", 100. * org_correct / total)
        print("Accuracy after attack: ", 100. * adv_correct / total)
        return adv_images


def main(args):
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    net = getattr(models, args.net_name)(pretrained=True)
    net = net.to(device)
    net.eval()
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Construct dataloader
    eval_dataset = EvalDataset(args.data_dir)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False)

    # Call attacker
    attacker = FGSM_Attacker(net, eval_loader)
    adv_imgs = attacker.attack(epsilon=args.epsilon,
                               max_iter=args.iter,
                               decode=True)

    # Save adv images
    save_dir = args.save_dir
    _save_makedirs(save_dir)
    for i_img in range(len(eval_dataset)):
        Im = Image.fromarray(adv_imgs[i_img])
        Im.save(os.path.join(save_dir, "%03d.png" % i_img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="../adv_imgs")

    parser.add_argument(
        "--net_name",
        type=str,
        default="vgg16",
        choices=["vgg16", "vgg19",
                 "resnet50", "resnet101",
                 "densenet121", "densenet169"]
        )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05)

    parser.add_argument(
        "--iter",
        type=int,
        default=1)

    args = parser.parse_args()

    main(args)
