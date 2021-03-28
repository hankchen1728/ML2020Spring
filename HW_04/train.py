import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import utils
import model
from data_loader import get_train_valid_dataloader


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def adjust_learning_rate(optimizer, epoch, initial_lr=0.1):
    """Sets the learning rate to the initial LR
    decayed by 0.95 each epoch
    """
    lr = initial_lr * (0.95 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluation(outputs, labels, threshold=0.5):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= threshold] = 1
    outputs[outputs < threshold] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


def trainer(net,
            train_loader,
            valid_loader,
            optimizer,
            criterion,
            epochs=100,
            initial_lr=1e-3,
            ckpt_dir="./checkpoint/GRU",
            device="cpu"):
    """Train progress function"""
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("\nstart training, parameter total:{}, "
          "trainable:{}\n".format(total, trainable))

    history = {"acc": [0.0] * epochs,
               "loss": [0.0] * epochs,
               "val_acc": [0.0] * epochs,
               "val_loss": [0.0] * epochs
               }

    _save_makedirs(ckpt_dir)
    best_val_acc = 0.0
    best_val_loss = 1.0
    for epoch in range(epochs):
        # Learing rate decay
        adjust_learning_rate(optimizer, epoch, initial_lr)
        epoch_start_time = time.time()
        train_loss = 0
        correct = 0
        total = 0
        # Training
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += evaluation(outputs, targets)

            utils.progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (train_loss/(batch_idx+1),
                   100.*correct/total, correct, total)
                )

        # Validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs = inputs.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)
                outputs = net(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_total += targets.size(0)
                val_correct += evaluation(outputs, targets)

        train_acc = 100. * correct / total
        train_loss /= train_loader.__len__()
        val_acc = 100. * val_correct / val_total
        val_loss /= valid_loader.__len__()
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
        # if (val_acc > best_val_acc) or (val_loss < best_val_loss):
        best_val_acc = max(best_val_acc, val_acc)
        best_val_loss = min(best_val_loss, val_loss)
        state = {
            'net': net.state_dict(),
            'acc': train_acc,
            'val_acc': val_acc,
            'epoch': epoch+1,
        }
        torch.save(
            state,
            os.path.join(ckpt_dir, "model_{:03}.pth".format(epoch+1))
            )

    # Save training history
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as opf:
        json.dump(history, opf)

    # End


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loader
    print("Constructing data loader ...")

    train_loader, valid_loader, embedding = get_train_valid_dataloader(
        train_fpath=args.train_fpath,
        train_nolabel_fpath=args.train_nolabel_fpath,
        w2v_path=args.w2v_path,
        sen_len=args.max_sen_len,
        batch_size=256,
        valid_split=0.15,
        num_workers=16,
        rand_seed=10,
        return_embedding=True)

    # Set network
    net = model.GRUNet(
        embedding=embedding,
        hidden_dim=512,
        num_layers=2,
        dropout_rate=0.4,
        fix_embedding=True
        )
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Set loss and optimizer
    initial_lr = args.lr
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), initial_lr)

    ckpt_dir = os.path.join(args.ckpt_dir, "GRU_Net")

    trainer(net,
            train_loader,
            valid_loader,
            optimizer,
            criterion,
            epochs=args.epochs,
            initial_lr=initial_lr,
            ckpt_dir=ckpt_dir,
            device=device)

    # End


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--train_fpath",
            type=str,
            default="./data/training_label.txt")

    parser.add_argument(
            "--train_nolabel_fpath",
            type=str,
            default="./data/training_nolabel.txt")

    parser.add_argument(
            "--ckpt_dir",
            type=str,
            default="./checkpoint")

    parser.add_argument(
            "--w2v_path",
            type=str,
            default="./checkpoint/w2v_all_250.model",
            help="File path to word2vector model")

    parser.add_argument(
            "--max_sen_len",
            type=int,
            default=20,
            help="Max sentence length for padding")

    parser.add_argument(
            "--lr",
            type=float,
            default=5e-4,
            help="learning rate")

    parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="training epochs")

    args = parser.parse_args()

    main(args)
