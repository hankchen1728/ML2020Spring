import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import model
from data_loader import get_test_dataloader


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def tester(net,
           test_loader,
           threshold=0.5,
           device="cpu"):
    """Predict progress function"""
    net.eval()
    with torch.no_grad():
        test_label = np.array([])
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = net(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= threshold] = 1
            outputs[outputs < threshold] = 0

            test_label = np.concatenate([test_label,
                                         outputs.cpu().data.numpy()])

    return test_label
    # End


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loader
    print("Constructing data loader ...")
    test_loader, embedding = get_test_dataloader(
        test_fpath=args.test_fpath,
        w2v_path=args.w2v_path,
        sen_len=args.max_sen_len,
        batch_size=256,
        return_embedding=True)

    # Set network
    net = model.GRUNet(
        embedding=embedding,
        hidden_dim=512,
        num_layers=2,
        dropout_rate=0.4,
        fix_embedding=True
        )

    checkpoint = torch.load(args.ckpt_path)
    print("Epoch: {}, Val acc: {}".format(checkpoint["epoch"],
                                          checkpoint["val_acc"]))
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.load_state_dict(checkpoint["net"], False)
    net.eval()

    test_pred = tester(net,
                       test_loader,
                       threshold=0.5,
                       device=device)
    # Save prediction
    pred_df = pd.DataFrame({
        "id": np.arange(test_pred.shape[0]),
        "label": test_pred.astype(dtype=np.int)
        })

    pred_df.to_csv(args.save_path, index=False)
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
            "--test_fpath",
            type=str,
            default="./data/testing_data.txt")

    parser.add_argument(
            "--ckpt_path",
            type=str,
            default="./checkpoint/GRU_Net/model_006.pth")

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
            "--save_path",
            type=str,
            default="./pred.csv")

    args = parser.parse_args()

    main(args)
