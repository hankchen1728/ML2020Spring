import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import models

import efficientNet
import data_loader
from net import StudentNet
from train_util import Evaluator, _save_makedirs
from weight_quantization import encode16bit


def config_net(net_name: str, num_classes=11):
    if net_name == "EfficientNet":
        net = efficientNet.EfficientNet(
            width_coefficient=0.3,
            depth_coefficient=0.4,
            image_size=128,
            dropout_rate=0.2,
            in_channels=3,
            num_classes=num_classes)
        image_size = net.image_size
    elif net_name == "MobileNetV2":
        inverted_residual_setting = [
            [1, 16, 1, 2],
            [6, 32, 2, 2],
            [6, 64, 2, 2],
            [4, 128, 1, 2]
        ]
        net = models.MobileNetV2(
            num_classes=11,
            inverted_residual_setting=inverted_residual_setting)
        net.features = nn.Sequential(*list(net.features.children())[:-1])
        net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 11),
            )
        image_size = (256, 256)
    elif net_name == "StudentNet":
        net = StudentNet(base=16)
        image_size = (256, 256)
    else:
        raise NotImplementedError("Unknown model architecture `%s`" % net_name)
    return net, image_size


def adjust_learning_rate(optimizer, epoch, initial_lr=0.1):
    """Sets the learning rate to the initial LR
    decayed by 0.97 every 3 epochs
    """
    lr = initial_lr * (0.97 ** (epoch / 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set network
    net, image_size = config_net(args.net_name, num_classes=11)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Set loss and optimizer
    initial_lr = args.lr
    optimizer = optim.Adam(net.parameters(), initial_lr)

    # Data loader
    print("Constructing data loader ...")
    data_dir = args.data_dir
    image_size = tuple(image_size)
    train_dir = os.path.join(data_dir, "training")
    train_loader = data_loader.get_train_dataloader(
        img_dir=train_dir,
        image_size=image_size,
        batch_size=32,
        num_workers=16,
        augment=True,
        shuffle=True,
        normalize=False
    )

    valid_dir = os.path.join(data_dir, "validation")
    valid_loader = data_loader.get_train_dataloader(
        img_dir=valid_dir,
        image_size=image_size,
        batch_size=256,
        num_workers=16,
        augment=False,
        shuffle=False,
        normalize=False
    )

    # Training processes
    epochs = args.epochs
    history = {"acc": [0.0] * epochs,
               "loss": [0.0] * epochs,
               "val_acc": [0.0] * epochs,
               "val_loss": [0.0] * epochs
               }

    ckpt_dir = os.path.join(args.ckpt_dir, args.net_name + "_fromScratch")
    _save_makedirs(ckpt_dir)

    # Initialize the evaluator
    best_param = None
    evaluator = Evaluator(net, optimizer, device, verbose=True)
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Learing rate decay
        # adjust_learning_rate(optimizer, epoch, initial_lr)
        epoch_start_time = time.time()

        train_acc, train_loss = evaluator.run_epoch(train_loader, "train")
        val_acc, val_loss = evaluator.run_epoch(valid_loader, "eval")

        print("[%03d/%03d] %2.2f sec(s)"
              " Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f" %
              (epoch + 1, epochs,
               time.time() - epoch_start_time,
               train_acc, train_loss, val_acc, val_loss
               ))

        history["acc"][epoch] = train_acc
        history["loss"][epoch] = train_loss
        history["val_acc"][epoch] = val_acc
        history["val_loss"][epoch] = val_loss

        # Save model checkpoints
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                'net': net.state_dict(),
                'acc': train_acc,
                'val_acc': val_acc,
                'epoch': epoch,
            }
            torch.save(
                state,
                os.path.join(ckpt_dir, "model_best.pth")
                )
            best_param = net.state_dict()

    # Apply weight quantization
    encode16bit(best_param, os.path.join(ckpt_dir, "model_WQ.npz"))

    # Save training history
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as opf:
        json.dump(history, opf)

    # end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Food-11 Training")

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--ckpt_dir",
            type=str,
            default="./checkpoints")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/food-11/")

    parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate")

    parser.add_argument(
            "--net_name",
            type=str,
            default="MobileNetV2",
            choices=["EfficientNet", "MobileNetV2", "StudentNet"])

    parser.add_argument(
            "--epochs",
            type=int,
            default=200,
            help="training epochs")

    args = parser.parse_args()

    main(args)
