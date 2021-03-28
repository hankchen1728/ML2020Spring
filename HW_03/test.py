import os
# import time
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import model
import data_loader


__all_models__ = ["EfficientNetB" + str(i) for i in range(0, 8)]


def config_net(num_classes):
    # assert net_name in __all_models__, "Unimplemented architecture!"
    # net = getattr(model, net_name)(in_channels=3, classes=num_classes)
    # image_size = net.image_size
    net = model.EfficientNet(width_coefficient=1.2,
                             depth_coefficient=1,
                             image_size=300,
                             dropout_rate=0.3,
                             in_channels=3,
                             num_classes=num_classes)
    image_size = net.image_size
    return net, image_size


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_prediction(pred_result, output_fpath):
    with open(output_fpath, 'w') as f:
        f.write('Id,Category\n')
        for idx, pred_value in enumerate(pred_result):
            f.write('%d,%d\n' % (idx, pred_value))


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set network and load weights
    net, image_size = config_net(num_classes=11)
    checkpoint = torch.load(args.ckpt_path)
    print("Epoch: {}, Val acc: {}".format(checkpoint["epoch"],
                                          checkpoint["val_acc"]))

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.load_state_dict(checkpoint['net'], False)
    net.eval()

    # Data loader
    print("Constructing data loader ...")
    data_dir = args.data_dir
    image_size = tuple(image_size)
    test_dir = os.path.join(data_dir, "testing")
    test_loader = data_loader.get_test_dataloader(
        img_dir=test_dir,
        image_size=image_size,
        batch_size=128,
        num_workers=16
    )

    test_label = np.array([])
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            test_label = np.concatenate([test_label,
                                         predicted.cpu().data.numpy()])

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
            "--ckpt_path",
            type=str,
            default="./model_weights/EfficientNetB3_best.pth")

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
