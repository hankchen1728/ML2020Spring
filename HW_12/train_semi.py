import os
import time
import argparse
import shutil
import json
import numpy as np
import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from models import (FeatureExtractor, LabelPredictor)
from dataloader import config_source_dataloader, config_target_dataloader
from train_util import (DaNN_Evaluator, get_highest_confidence, _save_makedirs)


def adjust_learning_rate(optimizer, epoch, initial_lr=0.1):
    """Sets the learning rate to the initial LR
    decayed by 0.97 every 3 epochs
    """
    lr = initial_lr * (0.97 ** (epoch / 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pred_label_confidence_score(
        feature_extractor, label_predictor,
        test_dataloader, device="cuda"):
    label_pred = list()
    confid_pred = list()
    with torch.no_grad():
        for batch_idx, (target_data, _) in enumerate(test_dataloader):
            target_data = target_data.to(device)
            class_logits = label_predictor(feature_extractor(target_data))
            confidences = torch.softmax(class_logits, dim=1)

            confidence, label = torch.max(confidences, dim=1)
            label_pred.append(label.cpu().detach().numpy())
            confid_pred.append(confidence.cpu().detach().numpy())

    label_pred = np.concatenate(label_pred, axis=0)
    confid_pred = np.concatenate(confid_pred, axis=0)
    return label_pred, confid_pred


def build_semi_target_dataset(
        label_pred: np.ndarray, confid_pred: np.ndarray,
        test_dir="./data/real_or_drawing/test_data/",
        dst_dirname="semi_test_data"):
    print("Now copy the image files for semi training")
    data_base_dir = os.path.dirname(test_dir.rstrip(os.sep))
    # Run throught all 10 classes
    for target_label in range(10):
        # Get the image indices with highest confidence
        target_indices = get_highest_confidence(
            label_pred, confid_pred,
            num_select=2500, target_label=target_label)
        # Make the directory
        target_dir = os.path.join(
            data_base_dir, dst_dirname, str(target_label)
            )
        _save_makedirs(target_dir)
        for img_idx in tqdm.tqdm(target_indices):
            img_fname = "%05d.bmp" % img_idx
            src_fpath = os.path.join(test_dir, '0', img_fname)
            dst_fpath = os.path.join(target_dir, img_fname)
            shutil.copyfile(src_fpath, dst_fpath)


def plot_classes_histogram(label_pred, save_fig=""):
    bins = np.arange(11) - 0.5
    plt.hist(label_pred, bins=bins, linewidth=1)
    plt.xticks(range(10))
    plt.xlim([-1, 10])
    plt.ylim([0, 12500])
    plt.plot(bins, [10000] * 11, color='red', linestyle='-')
    plt.savefig(save_fig)
    plt.clf()
    # end


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

    # Load model weights
    fe_ckpt = torch.load(os.path.join(ckpt_dir, "extractor.pth"))
    feature_extractor.load_state_dict(fe_ckpt["net"])
    feature_extractor.eval()

    lp_ckpt = torch.load(os.path.join(ckpt_dir, "classifier.pth"))
    label_predictor.load_state_dict(lp_ckpt["net"])
    label_predictor.eval()

    # Data loader
    print("Constructing data loader ...")
    train_dir = os.path.join(args.data_dir, "train_data")
    test_dir = os.path.join(args.data_dir, "test_data")
    semi_dirname = "semi_test_data"
    semi_dir = os.path.join(args.data_dir, semi_dirname)
    # source_train_dataloader, source_valid_dataloder = \
    #     config_source_dataloader(
    #         train_dir, batch_size=128, valid_size=0.1,
    #         shuffle=True, random_seed=20)
    source_train_dataloader = config_source_dataloader(
        train_dir, batch_size=256, valid_size=0,
        shuffle=True, random_seed=20)
    test_dataloader = config_target_dataloader(
        test_dir, batch_size=256, augment=False, shuffle=False)

    # Predict the label on target dataset
    label_pred, confidence_pred = pred_label_confidence_score(
        feature_extractor, label_predictor,
        test_dataloader, device=device)

    build_semi_target_dataset(
        label_pred, confidence_pred,
        test_dir=test_dir, dst_dirname=semi_dirname)

    semi_dataloader = config_target_dataloader(
        semi_dir, batch_size=256, augment=True, shuffle=True)

    # Training processes
    epochs = args.epochs
    history = {"loss": [0.0] * epochs, "acc": [0.0] * epochs}

    evaluator = DaNN_Evaluator(
        feature_extractor, label_predictor, label_predictor,
        device=device, verbose=True)
    initial_lr = args.lr
    evaluator.set_optimizer(opt_name="sgd", lr=initial_lr)
    # best_loss, best_val_acc = float("inf"), 0
    _save_makedirs("./pred_histograms/semi")

    for epoch in range(epochs):
        # epoch_start_time = time.time()
        loss, acc = evaluator.finetune_epoch(
            source_train_dataloader, semi_dataloader)
        # source_val_acc = evaluator.pred_acc(source_valid_dataloder)

        history["loss"][epoch] = loss
        history["acc"][epoch] = acc

        # Plot the prediction
        # test_label_pred, _ = pred_label_confidence_score(
        #     feature_extractor, label_predictor, test_dataloader, device
        #     )
        # plot_classes_histogram(
        #     test_label_pred,
        #     os.path.join("./pred_histograms/semi", "%03d.png" % (epoch+1))
        #     )
        # print("[%03d/%03d] %2.2f sec(s)"
        #       " Training Mix acc: %.4f | Source Val acc: %.4f" %
        #       (epoch + 1, epochs, time.time() - epoch_start_time,
        #        acc, source_val_acc)
        #       )

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
            default="./checkpoints/")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/real_or_drawing/")

    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="learning rate")

    parser.add_argument(
            "--epochs",
            type=int,
            default=200,
            help="training epochs")

    args = parser.parse_args()

    main(args)
