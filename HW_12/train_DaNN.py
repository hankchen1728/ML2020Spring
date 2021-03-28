import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import (FeatureExtractor, LabelPredictor, DomainClassifier)
from dataloader import config_source_dataloader, config_target_dataloader
from train_util import (DaNN_Evaluator, _save_makedirs)
from train_semi import plot_classes_histogram, pred_label_confidence_score


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
    domain_classifier = DomainClassifier().to(device)

    if device == 'cuda':
        feature_extractor = nn.DataParallel(feature_extractor)
        label_predictor = nn.DataParallel(label_predictor)
        domain_classifier = nn.DataParallel(domain_classifier)
        cudnn.benchmark = True

    # Set loss and optimizer
    print("Setting optimizer...")
    initial_lr = args.lr

    # Data loader
    print("Constructing data loader ...")
    train_dir = os.path.join(args.data_dir, "train_data")
    test_dir = os.path.join(args.data_dir, "test_data")
    # source_train_dataloader, source_valid_dataloder = \
    #     config_source_dataloader(
    #         train_dir, batch_size=128, valid_size=0.1,
    #         shuffle=True, random_seed=10)
    source_train_dataloader = config_source_dataloader(
        train_dir, batch_size=128, valid_size=0,
        shuffle=True, random_seed=10)
    target_dataloader = config_target_dataloader(
        test_dir, batch_size=128, augment=True, shuffle=True)
    test_dataloader = config_target_dataloader(
        test_dir, batch_size=256, augment=False, shuffle=False)

    # Training processes
    epochs = args.epochs
    history = {
        "domain_loss": [0.0] * epochs,
        "mix_loss": [0.0] * epochs,
        "source_acc": [0.0] * epochs
        }

    ckpt_dir = args.ckpt_dir
    _save_makedirs(ckpt_dir)

    evaluator = DaNN_Evaluator(
        feature_extractor, label_predictor, domain_classifier,
        device=device, verbose=True)
    evaluator.set_optimizer(opt_name="adam", lr=initial_lr)
    best_mix_loss = float("inf")
    # best_d_loss = float("inf")
    # First use adam for optimizer
    _save_makedirs("./pred_histograms/DaNN")
    for epoch in range(epochs):
        # epoch_start_time = time.time()
        domain_loss, mix_loss, source_acc = evaluator.train_epoch(
            source_train_dataloader, target_dataloader)

        # source_val_acc = evaluator.pred_acc(source_valid_dataloder)

        # print("[%03d/%03d] %2.2f sec(s)"
        #       " Source acc: %3.6f | Source Val acc: %3.6f" %
        #       (epoch + 1, epochs, time.time() - epoch_start_time,
        #        source_acc, source_val_acc)
        #       )
        history["domain_loss"][epoch] = domain_loss
        history["mix_loss"][epoch] = mix_loss
        history["source_acc"][epoch] = source_acc

        # Save model checkpoints
        if epoch % 20 == 0:
            print("Saving model weights epochs: %03d" % (epoch+1))
            # best_mix_loss = mix_loss
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

        if epoch % 100 == 0:
            # Plot the prediction
            test_label_pred, _ = pred_label_confidence_score(
                feature_extractor, label_predictor, test_dataloader, device
                )
            plot_classes_histogram(
                test_label_pred,
                os.path.join("./pred_histograms/DaNN", "%03d.png" % (epoch+1))
                )

        if epoch == 400:
            evaluator.set_optimizer("sgd", lr=initial_lr*0.1)

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
