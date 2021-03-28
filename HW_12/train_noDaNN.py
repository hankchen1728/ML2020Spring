import os
import argparse
import json
import numpy as np
import tqdm

import torch
import torch.nn as nn
from models import (FeatureExtractor, LabelPredictor)
from dataloader import config_source_dataloader
from train_util import (DaNN_Evaluator, _save_makedirs)


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
    print("Construct models...")
    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)

    if device == 'cuda':
        feature_extractor = nn.DataParallel(feature_extractor)
        label_predictor = nn.DataParallel(label_predictor)

    ckpt_dir = args.ckpt_dir
    _save_makedirs(ckpt_dir)

    # Data loader
    print("Constructing data loader ...")
    train_dir = os.path.join(args.data_dir, "train_data")

    source_train_dataloader = config_source_dataloader(
        train_dir, batch_size=256, valid_size=0,
        shuffle=True, random_seed=20)
    # Training processes
    epochs = args.epochs
    history = {"loss": [0.0] * epochs, "acc": [0.0] * epochs}

    evaluator = DaNN_Evaluator(
        feature_extractor, label_predictor, label_predictor,
        device=device, verbose=True)
    initial_lr = args.lr
    evaluator.set_optimizer(opt_name="adam", lr=initial_lr)
    # best_loss, best_val_acc = float("inf"), 0

    for epoch in range(epochs):
        # epoch_start_time = time.time()
        loss, acc = evaluator.finetune_epoch(
            source_train_dataloader, source_train_dataloader)
        # source_val_acc = evaluator.pred_acc(source_valid_dataloder)

        history["loss"][epoch] = loss
        history["acc"][epoch] = acc

        # Save model checkpoints
        print("Saving model weights epochs: %03d" % (epoch+1))
        # best_loss = loss
        # best_val_acc = source_val_acc
        fe_state = {
            'net': feature_extractor.state_dict(),
            'epoch': epoch
        }
        torch.save(fe_state, os.path.join(ckpt_dir, "extractor.pth"))
        lp_state = {
            'net': label_predictor.state_dict(),
            'epoch': epoch
        }
        torch.save(lp_state, os.path.join(ckpt_dir, "classifier.pth"))

    # Save training history
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as opf:
        json.dump(history, opf)
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
            "--ckpt_dir",
            type=str,
            default="./checkpoints/noDaNN/")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/real_or_drawing/")

    parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate")

    parser.add_argument(
            "--epochs",
            type=int,
            default=200,
            help="training epochs")

    args = parser.parse_args()

    main(args)
