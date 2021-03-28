import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from net import AutoEncoder
from net import BaselineEncoder
from data_loader import get_train_valid_dataLoader


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
    if args.improved:
        net = AutoEncoder()
    else:
        net = BaselineEncoder()
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
    train_loader, valid_loader = get_train_valid_dataLoader(
        trainX_npy,
        batch_size=32,
        valid_split=0.1,
        num_workers=16,
        rand_seed=random_seed)

    # Training processes
    epochs = args.epochs
    history = {"loss": [0.0] * epochs,
               "val_loss": [0.0] * epochs}

    ckpt_path = args.ckpt_path
    ckpt_dir = os.path.dirname(ckpt_path)
    _save_makedirs(ckpt_dir)
    best_val_loss = 1.0
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
        val_loss = 0.0
        net.eval()
        with torch.no_grad():
            for batch_idx, (data) in enumerate(valid_loader):
                data = data.to(device)
                _, reconst = net(data)
                loss = criterion(data, reconst)

                val_loss += loss.item()

        # Show the result of current epoch
        train_loss /= train_loader.__len__()
        val_loss /= valid_loader.__len__()
        print("[%03d/%03d] %2.2f sec(s)"
              " Train Loss: %3.6f | Val Loss: %3.6f" %
              (epoch + 1, epochs,
               time.time() - epoch_start_time,
               train_loss, val_loss
               ))

        history["loss"][epoch] = train_loss
        history["val_loss"][epoch] = val_loss

        # Save model checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                'net': net.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch + 1,
            }
            torch.save(state, ckpt_path)

    # Save training history
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as opf:
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
            "--ckpt_path",
            type=str,
            default="./checkpoints/baseline.path")

    parser.add_argument(
            "--improved",
            action="store_true",
            default=False)

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
            default=300,
            help="training epochs")

    args = parser.parse_args()

    main(args)
