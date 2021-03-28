import os
import argparse
import numpy as np

import util


def generate_data(test_data):
    test_size = 240
    test_x = np.empty([test_size, 20, 9], dtype=float)
    for i in range(test_size):
        test_x[i, :18] = test_data[18 * i: 18 * (i + 1), :]
        util.fix_negative_feature(test_x[i, :], i_feature=9)

        wind_direct = test_x[i, 15]
        cos_WD = np.cos(wind_direct)[np.newaxis, :]
        sin_WD = np.sin(wind_direct)[np.newaxis, :]
        test_x[i, 18] = cos_WD
        test_x[i, 19] = sin_WD

    return test_x


def save_result(output_fpath, pred_resulf):
    with open(output_fpath, 'w') as opf:
        opf.write("id,value\n")
        for idx in range(pred_resulf.shape[0]):
            opf.write("id_{},{}\n".format(idx, pred_resulf[idx][0]))


def main(args):
    test_df = util.read_df(args.test_csv)
    test_data = test_df.iloc[:, 2:].to_numpy()

    test_x = generate_data(test_data)

    # Load normalization mean and std
    normalize_weights = np.load(
            os.path.join(args.model_path, "norm_param.npz")
            )
    feature_mean = normalize_weights["mean"]
    feature_std = normalize_weights["std"]

    test_x = (test_x - feature_mean) / feature_std
    test_x = np.reshape(test_x, newshape=(test_x.shape[0], -1))
    test_x = util.append_const(test_x)

    # Load model weights and predict
    weight = np.load(
            os.path.join(args.model_path, "weight.npz")
            )["w"]

    pred_y = np.dot(test_x, weight)
    # fix negative value
    pred_y[pred_y < 0] = 0

    save_result(args.output_file, pred_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--test_csv",
            type=str,
            default="./data/train.csv")

    parser.add_argument(
            "--output_file",
            type=str,
            default="./submit.csv")

    parser.add_argument(
            "--model_path",
            type=str,
            default="./model")

    args = parser.parse_args()
    main(args)
