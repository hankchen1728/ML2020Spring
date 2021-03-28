import os
import argparse
from GAN_model import Generator

import torch
from torch.autograd import Variable
import torchvision


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model and load weights
    G = Generator(args.z_dim).to(device)
    G.load_state_dict(torch.load(args.gen_ckpt))
    G.eval()

    same_seeds(10)
    n_output = 20
    z_sample = Variable(torch.randn(n_output, args.z_dim)).to(device)
    imgs_sample = (G(z_sample).data + 1) / 2.0
    save_img_fpath = args.save_img
    save_base_dir = os.path.dirname(save_img_fpath)
    os.makedirs(save_base_dir, exist_ok=True)
    torchvision.utils.save_image(imgs_sample, save_img_fpath, nrow=10)
    # end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--z_dim",
            type=int,
            default=100,
            help="GAN z dimensions")

    parser.add_argument(
            "--gen_ckpt",
            type=str,
            default="./checkpoints/GAN/dcgan_g.pth")

    parser.add_argument(
            "--save_img",
            type=str,
            default="./image_rep/p1.png")

    args = parser.parse_args()

    main(args)
