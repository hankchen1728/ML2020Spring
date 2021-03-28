import numpy as np


def normalize(
        X,
        train=True,
        specified_column=None,
        X_mean=None, X_std=None, keep=True):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused
    # when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized.
    #         If 'None', all columns will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column is None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    norm_X = X if keep else X.copy()
    norm_X[:, specified_column] = \
        (norm_X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return norm_X, X_mean, X_std


def preprocessing(
        X, Train=True,
        X_train_mean=None, X_train_std=None,
        feature_mask=None, add_square_and_cubic=True):
    continue_col = [0, 126, 210, 211, 212, 358, 507]

    if Train:
        X_norm, X_train_mean, X_train_std = normalize(
                X,
                train=True,
                specified_column=continue_col)
    else:
        assert (X_train_mean is not None) and (X_train_std is not None)
        X_norm, _, _ = normalize(
                X,
                train=False,
                specified_column=continue_col,
                X_mean=X_train_mean,
                X_std=X_train_std)

    Square_continue = np.power(X_norm[:, continue_col], 2)
    Cubic_continue = np.power(X_norm[:, continue_col], 3)
#     Exp_continue = np.exp(X_norm[:, continue_col])
#     Exp_continue = (Exp_continue - np.mean(Exp_continue, axis = 0)) \
#         / np.std(Exp_continue, axis = 0)
#     Root_continue = np.sqrt(X_continue)
#     Root_continue = (Root_continue - np.mean(Root_continue, axis = 0)) \
#         / np.std(Root_continue, axis = 0)

    if add_square_and_cubic:
        if feature_mask is None:
            X_select = np.concatenate(
                    (X_norm,
                     Square_continue,
                     Cubic_continue
                     ), axis=1)
        else:
            X_select = np.concatenate(
                    (X_norm[:, feature_mask],
                     Square_continue,
                     Cubic_continue
                     ), axis=1)
    else:
        X_select = X_norm[:, feature_mask]

    if Train:
        return X_select, X_train_mean, X_train_std
    return X_select


def split_train_val(X, y, val_size=0.1, shuffle=True, rand_seed=10):
    num = X.shape[0]
    val_num = int(num * val_size)

    sample_idx = np.arange(num)
    if shuffle:
        # Fix random seed
        np.random.seed(rand_seed)
        np.random.shuffle(sample_idx)

    return (X[sample_idx[val_num:]],
            y[sample_idx[val_num:]],
            X[sample_idx[: val_num]],
            y[sample_idx[: val_num]])


def save_result(output_fpath, pred_resulf):
    with open(output_fpath, 'w') as f:
        f.write('id,label\n')
        for idx, pred_value in enumerate(pred_resulf):
            f.write('%d,%d\n' % (idx, pred_value))
