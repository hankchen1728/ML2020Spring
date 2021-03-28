import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from net import AutoEncoder
from data_loader import get_train_valid_dataLoader
from image_cluster import clustering, image_encoding, cal_acc


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def torch_rand_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, initial_lr=0.1):
    """Sets the learning rate to the initial LR
    decayed by 0.9 every 3 epochs
    """
    lr = initial_lr * (0.9 ** (epoch / 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random_seed = args.random_seed
    # Set network
    torch_rand_seeds(random_seed)
    net = AutoEncoder()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)

    # Set loss and optimizer
    initial_lr = args.lr
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        net.parameters(),
        initial_lr,
        weight_decay=1e-5)

    # Data loader
    print("Constructing data loader ...")
    trainX_npy = args.trainX_fpath
    train_loader = get_train_valid_dataLoader(
        trainX_npy,
        batch_size=64,
        valid_split=0,
        num_workers=16,
        rand_seed=random_seed)

    valX_npy = args.valX_fpath
    val_label = np.load(args.valY_fpath)

    # Training processes
    epochs = args.epochs
    history = {"loss": [0.0] * epochs,
               "all_loss": [0.0] * epochs,
               "val_acc": [0.0] * epochs}

    ckpt_path = args.ckpt_path
    ckpt_dir = os.path.dirname(ckpt_path)
    _save_makedirs(ckpt_dir)
    # best_val_loss = 1.0
    for epoch in range(epochs):
        # Learing rate decay
        # adjust_learning_rate(optimizer, epoch, initial_lr)
        epoch_start_time = time.time()
        train_loss = 0.0
        # Training
        net.train()
        for batch_idx, (data) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            _, reconst = net(data)
            loss = criterion(data, reconst)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # utils.progress_bar(
            #     batch_idx,
            #     len(train_loader),
            #     "Loss: %.3f" % (train_loss/(batch_idx+1))
            #     )

        # Validation
        train_all_loss = 0.0
        net.eval()
        with torch.no_grad():
            for batch_idx, (data) in enumerate(train_loader):
                data = data.to(device)
                _, reconst = net(data)
                loss = criterion(data, reconst)

                train_all_loss += loss.item()

        # Get validation encoding
        valid_encoding = image_encoding(
            valX_npy, net, device="cuda", batch_size=128)
        valid_cluster, _ = clustering(
                valid_encoding,
                pca_ndim=100,
                random_seed=10)
        val_acc = cal_acc(val_label, valid_cluster)

        # Show the result of current epoch
        train_loss /= train_loader.__len__()
        train_all_loss /= train_loader.__len__()
        print("[%03d/%03d] %2.2f sec(s)"
                " Train Loss: %3.6f | Train all Loss: %3.6f | Val Acc: %f" %
              (epoch + 1, epochs,
               time.time() - epoch_start_time,
               train_loss, train_all_loss, val_acc
               ))

        history["loss"][epoch] = train_loss
        history["all_loss"][epoch] = train_all_loss
        history["val_acc"][epoch] = val_acc

        # Save model checkpoints
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     state = {
        #         'net': net.state_dict(),
        #         'val_loss': val_loss,
        #         'epoch': epoch + 1,
        #     }
        #     torch.save(state, ckpt_path)

    # Save training history
    with open(os.path.join(ckpt_dir, "report_history.json"), 'w') as opf:
        json.dump(history, opf)

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
            "--trainX_fpath",
            type=str,
            default="./data/trainX.npy")

    parser.add_argument(
            "--valX_fpath",
            type=str,
            default="./data/valX.npy")

    parser.add_argument(
            "--valY_fpath",
            type=str,
            default="./data/valY.npy")

    parser.add_argument(
            "--ckpt_path",
            type=str,
            default="./checkpoints/baseline.pth")

    parser.add_argument(
            "--random_seed",
            type=int,
            default=10)

    parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate")

    parser.add_argument(
            "--epochs",
            type=int,
            default=20,
            help="training epochs")

    args = parser.parse_args()

    main(args)
