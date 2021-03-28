import os
import argparse
import time

import torch
import numpy as np
import pandas as pd

from models import (FeatureExtractor, LabelPredictor)
from dataloader import config_target_dataloader
# from train_util import _save_makedirs


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = args.ckpt_dir
    # Set network
    print("Construct models...")
    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)

    if device == "cuda":
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        label_predictor = torch.nn.DataParallel(label_predictor)

    # Load model weights
    fe_ckpt = torch.load(os.path.join(ckpt_dir, "extractor.pth"))
    feature_extractor.load_state_dict(fe_ckpt["net"])
    feature_extractor.eval()

    lp_ckpt = torch.load(os.path.join(ckpt_dir, "classifier.pth"))
    label_predictor.load_state_dict(lp_ckpt["net"])
    label_predictor.eval()

    print("Epoch: %3d" % lp_ckpt['epoch'])
    # Data loader
    target_dir = os.path.join(args.data_dir, "test_data")
    test_dataloader = config_target_dataloader(
        target_dir, batch_size=128, augment=False, shuffle=False)

    print("start prediction")
    label_pred = list()
    with torch.no_grad():
        for batch_idx, (target_data, _) in enumerate(test_dataloader):
            target_data = target_data.to(device)
            class_logits = label_predictor(feature_extractor(target_data))

            label = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            label_pred.append(label)

    # Save prediction result to csv file
    label_pred = np.concatenate(label_pred, axis=0)
    result_df = pd.DataFrame(
        {'id': np.arange(label_pred.shape[0]), 'label': label_pred}
        )
    result_df.to_csv(args.save_path, index=False)
    # end


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="ML2020Spring HW12 test file"
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
            "--save_path",
            type=str,
            default="./submission.csv")

    args = parser.parse_args()

    main(args)
    print("Spent time: %4.6f sec(s)" % (time.time()-start_time))
