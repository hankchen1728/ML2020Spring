import os
import argparse
import random

import numpy as np
import torch
# import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

from data_loader import get_dataset
from WGAN_model import WGenerator, WDiscriminator


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

    # Build models
    z_dim = args.z_dim
    WG = WGenerator(in_dim=z_dim).to(device)
    WD = WDiscriminator(3).to(device)
    WG.train()
    WD.train()

    # optimizer
    opt_WD = optim.RMSprop(WD.parameters(), lr=args.lr)
    opt_WG = optim.RMSprop(WG.parameters(), lr=args.lr)

    # for logging
    z_sample = Variable(torch.randn(100, z_dim)).to(device)

    ckpt_dir = os.path.dirname(args.gen_ckpt)
    log_dir = args.log_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    clamp_num = args.clamp_num
    n_epochs = args.epochs
    for e, epoch in enumerate(range(n_epochs)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.to(device)

            bs = imgs.size(0)  # batch size

            for parm in WD.parameters():
                parm.data.clamp_(-clamp_num, clamp_num)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).to(device)  # noise samples
            r_imgs = Variable(imgs).to(device)  # real image
            f_imgs = WG(z)  # (fake) generated image

            # label
            r_label = torch.ones((bs)).to(device)

            # dis
            r_logit = WD(r_imgs.detach())
            f_logit = WD(f_imgs.detach())

            # compute loss
            V_D = np.sum(r_logit.cpu().data.numpy()) - \
                np.sum(f_logit.cpu().data.numpy())

            # update model
            WD.zero_grad()
            r_logit.backward(r_label)
            f_logit.backward(r_label*(-1))
            opt_WD.step()

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).to(device)
            f_imgs = WG(z)

            # dis
            f_logit = WD(f_imgs)

            # compute loss
            V_G = - np.sum(f_logit.cpu().data.numpy())

            # update model
            WG.zero_grad()
            f_logit.backward(r_label)
            opt_WG.step()

            # log
            print(f"\rEpoch [{epoch+1}/{n_epochs}] {i+1}/{len(dataloader)} Loss_WD: {V_D:.4f} Loss_WG: {V_G:.4f}",
                  end='')

        WG.eval()
        f_imgs_sample = (WG(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        WG.train()
        torch.save(WG.state_dict(), args.gen_ckpt)
        # torch.save(
        #     WD.state_dict(),
        #     os.path.join(ckpt_dir, "wgan_d.pth"))

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
            default="./checkpoints/p2_g.pth")

    parser.add_argument(
            "--log_dir",
            type=str,
            default="./log/WGAN")

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
            "--clamp_num",
            type=float,
            default=0.01,
            help="WGAN clamp num")

    parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="training epochs")

    args = parser.parse_args()

    main(args)
