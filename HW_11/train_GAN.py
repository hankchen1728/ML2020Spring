import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

from data_loader import get_dataset
from GAN_model import Generator, Discriminator


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    same_seeds(10)
    # Data loader
    print("Configure dataloader ...")
    dataset = get_dataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=8)

    # loss criterion
    criterion = nn.BCELoss()

    # Build models
    z_dim = args.z_dim
    G = Generator(in_dim=z_dim).to(device)
    D = Discriminator(3).to(device)
    G.train()
    D.train()

    # optimizer
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # for logging
    z_sample = Variable(torch.randn(100, z_dim)).to(device)

    ckpt_dir = os.path.dirname(args.gen_ckpt)
    log_dir = args.log_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    n_epochs = args.epochs
    for e, epoch in enumerate(range(n_epochs)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.to(device)

            bs = imgs.size(0)  # batch size

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).to(device)  # noise samples
            r_imgs = Variable(imgs).to(device)  # real image
            f_imgs = G(z)  # (fake) generated image

            # label
            r_label = torch.ones((bs)).to(device)
            f_label = torch.zeros((bs)).to(device)

            # dis
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # compute loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # update model
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).to(device)
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)

            # compute loss
            loss_G = criterion(f_logit, r_label)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print(f"\rEpoch [{epoch+1}/{n_epochs}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}",
                  end='')

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        G.train()
        # if (e+1) % 5 == 0:
        torch.save(G.state_dict(), args.gen_ckpt)
        # torch.save(
        #     D.state_dict(),
        #     os.path.join(ckpt_dir, "dcgan_d.pth"))
    # end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch AutoEncoder Training"
        )

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--gen_ckpt",
            type=str,
            default="./checkpoints/p1_g.pth")

    parser.add_argument(
            "--log_dir",
            type=str,
            default="./log/GAN")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/")

    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="learning rate")

    parser.add_argument(
            "--z_dim",
            type=int,
            default=100,
            help="GAN z dimensions")

    parser.add_argument(
            "--epochs",
            type=int,
            default=5,
            help="training epochs")

    args = parser.parse_args()

    main(args)
