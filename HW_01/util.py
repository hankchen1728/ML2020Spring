import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_df(csv_path):
    df = pd.read_csv(csv_path, header=None, encoding="big5")
    # change to "NR" to 0
    df[df == "NR"] = 0
    return df


def fix_negative_feature(month_feature, i_feature=9):
    """Fix negative value with previous hour value
    """
    sample_feature = month_feature[i_feature]
    neg_idices = np.where(sample_feature < 0)[0]

    for _i in range(neg_idices.size):
        curr_h = neg_idices[_i]
        if curr_h == 0:
            sample_feature[curr_h] = 0
        else:
            sample_feature[curr_h] = sample_feature[curr_h-1]


def append_const(X):
    x_size = X.shape[0]
    const_term = np.ones([x_size, 1])
    return np.concatenate([const_term, X.copy()], axis=1)


def plot_history(history_dict, record_step):
    train_error = history_dict["train"]
    val_error = history_dict["val"]
    n_epoch = train_error.shape[0]
    # plot loss
    plt.figure(figsize=(8, 6))
    xtick = np.arange(0, n_epoch) * record_step
    plt.plot(xtick, train_error, label="Train")
    plt.plot(xtick, val_error, label="Val")

    # Auxiliary line
    plt.plot(xtick, np.full((n_epoch), 5.5), '-.', color='0.8')
    plt.plot(xtick, np.full((n_epoch), 6), '-.', color='0.8')

    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, n_epoch * record_step, step=500))
    plt.ylim(top=7, bottom=5)
    plt.show()


def to_k_folds(elements: np.array, n_folds=4):
    """Split the list into k folds
    """
    elements = np.array(elements)
    n_elements = elements.shape[0]
    num_per_fold = n_elements // n_folds
    idx_list = list(range(0, n_elements, num_per_fold))[1: n_folds]

    element_folds = [elements[s: e] for s, e in
                     zip([0] + idx_list, idx_list + [None])]

    assert len(element_folds) == n_folds
    return element_folds


def CV_train_valid_split(n_samples, random_seed=20, n_folds=5):
    # Fix random seed
    np.random.seed(random_seed)
    # Shuffle the id list
    data_idices = np.arange(n_samples)
    np.random.shuffle(data_idices)

    id_folds = to_k_folds(data_idices, n_folds=n_folds)
    return id_folds
