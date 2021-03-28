import os
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import data_loader
from train_util import Evaluator
from train import config_net
from weight_quantization import decode16bit


def save_prediction(pred_result, output_fpath):
    with open(output_fpath, 'w') as f:
        f.write('Id,label\n')
        for idx, pred_value in enumerate(pred_result):
            f.write('%d,%d\n' % (idx, pred_value))


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set network and load weights
    net, image_size = config_net(net_name="MobileNetV2", num_classes=11)
    # checkpoint = torch.load("checkpoints/MobileNetV2/model_best.pth")

    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    param_state_dict = decode16bit(args.param_npz)
    net.load_state_dict(param_state_dict, False)
    # net.load_state_dict(checkpoint["net"])
    net.eval()

    # Data loader
    print("Constructing data loader ...")
    data_dir = args.data_dir
    image_size = tuple(image_size)
    test_dir = os.path.join(data_dir, "testing")
    test_loader = data_loader.get_test_dataloader(
        img_dir=test_dir,
        image_size=image_size,
        batch_size=256,
        num_workers=16,
        normalize=False
    )

    evaluator = Evaluator(net, None, device=device, verbose=False)
    test_label = evaluator.predict(test_loader)

    save_prediction(test_label, args.save_path)
    # end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--param_npz",
            type=str,
            default="./checkpoints/MobileNetV2/model_wq.npz")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/food-11/")

    parser.add_argument(
            "--save_path",
            type=str,
            default="./pred.csv")

    args = parser.parse_args()

    main(args)
