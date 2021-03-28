import os
import argparse
import numpy as np

import util


def normalization(x_data):
    feature_mean = np.mean(x_data, axis=(0, 2))
    feature_std = np.std(x_data, axis=(0, 2))

    feature_mean = feature_mean[:, np.newaxis]
    feature_std = feature_std[:, np.newaxis]

    norm_x = (x_data - feature_mean) / feature_std
    norm_x = np.reshape(norm_x, newshape=(norm_x.shape[0], -1))

    return norm_x, feature_mean, feature_std


def generate_data(raw_data):
    # Split raw data by "month" and deal with negative pm2.5 value
    months_dict = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            total_day = 20 * month + day
            sample[:, day * 24: (day + 1) * 24] = \
                raw_data[18 * total_day: 18 * (total_day + 1), :]

        # Fill negative pm2.5 with 0
        util.fix_negative_feature(sample, i_feature=9)
        # Process wind direction
        wind_direct = sample[15, :]
        cos_WD = np.cos(wind_direct)[np.newaxis, :]
        sin_WD = np.sin(wind_direct)[np.newaxis, :]
        sample = np.concatenate([sample, cos_WD, sin_WD])

        months_dict[month] = sample

    # Generating X and Y
    n_feature = 20
    x = np.empty([12 * 471, n_feature, 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        month_data = months_dict[month]
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                data_idx = month * 471 + day * 24 + hour
                curr_hour = day * 24 + hour
                x[data_idx] = month_data[:, curr_hour: curr_hour + 9]
                y[data_idx, 0] = month_data[9, curr_hour + 9]

    return x, y


def rmse(X, weights, y):
    y_pred = np.dot(X, weights)
    return np.sqrt(np.mean(np.power(y_pred - y, 2)))


def training_adagrad(
        train_X,
        train_y,
        epoch=1000,
        init_lr=1,
        Val_X=None,
        Val_y=None,
        regular_lambda=1e-3,
        record_step=50,
        plotting=True):
    # Initialize weight
    n_features = train_X.shape[1] + 1
    weight = np.zeros(shape=(n_features, 1))
    # weight = np.random.normal(loc=0, scale=1, size=n_features)

    # learning_rate = np.full(shape=(n_features), fill_value=init_lr)
    learning_rate = init_lr

    # concate constant term
    # train_size = train_X.shape[0]
    train_X = util.append_const(train_X)

    train_error = np.zeros(shape=(epoch // record_step + 1))
    train_error[0] = rmse(train_X, weight, train_y)

    # Check validation set
    is_val = not (Val_X is None or Val_y is None)
    val_error = None
    # val_size = 0
    if is_val:
        # val_size = Val_X.shape[0]
        Val_X = util.append_const(Val_X)

        val_error = np.zeros(shape=(epoch // record_step + 1))
        val_error[0] = rmse(Val_X, weight, Val_y)

    ada = np.zeros(shape=(n_features, 1))
    # start gradient descent
    for i in range(1, epoch + 1):
        y_pred = np.dot(train_X, weight)
        # regularization
        grad = 2 * (np.dot(train_X.T, y_pred - train_y)
                    + regular_lambda * weight)

        ada += grad ** 2
        weight -= learning_rate * grad / np.sqrt(ada + 1e-8)

        # compute error
        if (i % record_step) == 0:
            i_record = i // record_step
            train_error[i_record] = rmse(train_X, weight, train_y)

            if is_val:
                val_error[i_record] = rmse(Val_X, weight, Val_y)

    print("Final training loss:", train_error[-1])
    # plot loss
    if is_val:
        print("Final validation loss:", val_error[-1])

    history_dict = {"train": train_error, "val": val_error}
    if plotting:
        util.plot_history(history_dict, record_step=50)

    return weight, history_dict


def main(args):
    train_df = util.read_df(args.train_csv)
    raw_data = train_df.iloc[:, 3:].to_numpy()

    # Create directory for saving model weights
    util._save_makedirs(args.model_path)

    all_X, all_y = generate_data(raw_data)

    # Normalization, saving normalization mean and std.
    norm_X, feature_mean, feature_std = normalization(all_X)

    np.savez(
        os.path.join(args.model_path, "norm_param.npz"),
        mean=feature_mean,
        std=feature_std
        )

    # Apply cross validation
    id_folds = util.CV_train_valid_split(
            all_y.shape[0],
            random_seed=20,
            n_folds=5)

    # Choose 5-fold `02`
    i_fold = 2
    valid_ids = id_folds[i_fold]
    train_ids = np.concatenate(
        [id_folds[_j] for _j in range(len(id_folds)) if _j != i_fold],
        axis=0)

    train_X, train_y = norm_X[train_ids], all_y[train_ids]
    val_X, val_y = norm_X[valid_ids], all_y[valid_ids]

    weight, history_dict = training_adagrad(
        train_X=train_X,
        train_y=train_y,
        epoch=int(5000),
        init_lr=50,
        Val_X=val_X,
        Val_y=val_y,
        regular_lambda=1e-3,
        plotting=False)

    np.savez_compressed(
            os.path.join(args.model_path, "weight.npz"),
            weight=weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--train_csv",
            type=str,
            default="./data/train.csv")

    parser.add_argument(
            "--model_path",
            type=str,
            default="./model")

    args = parser.parse_args()
    main(args)
