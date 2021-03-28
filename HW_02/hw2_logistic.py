import os
import sys
import numpy as np

import tqdm
import util


def Parse_argv(argv_list):
    To_train = False
    if len(argv_list) > 5 and argv_list[5] == "--Train":
        To_train = True
    X_train_fpath = argv_list[1]
    Y_train_fpath = argv_list[2]
    X_test_fpath = argv_list[3]
    Pred_save_fpath = argv_list[4]

    return X_train_fpath, Y_train_fpath, X_test_fpath, Pred_save_fpath, To_train


def drop_useless(X):
    # drop fnlwgt and other useless feature
    country_col = list(range(64, 106))

    To_drop_col = [1] + country_col
    X_reduce = np.delete(X, To_drop_col, axis=1)

    USA = np.reshape(X[:, 102], (X.shape[0], 1))

    X_reduce = np.concatenate((X_reduce, USA), axis=1)
    return X_reduce


# Logistic regression used functions
def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def forward(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def cross_entropy(y_pred, y_true):
    CE = - np.dot(y_true, np.log(y_pred))\
        - np.dot((1 - y_true), np.log(1 - y_pred))
    return CE


def compute_gradient(X, y_true, w, b, reg_lambda=0):
    y_pred = forward(X, w, b)
    diff = y_true - y_pred
    w_grad = -np.mean(np.multiply(diff.T, X.T), 1) + reg_lambda * w
    b_grad = -np.mean(diff)
    return w_grad, b_grad


def compute_loss_and_acc(X, y_true, w, b, reg_lambda=0):
    dataNum = y_true.shape[0]
    y_pred = forward(X, w, b)
    loss = (cross_entropy(y_pred, y_true)
            + reg_lambda * np.sum(np.square(w))) / dataNum

    y_pred_class = np.round(y_pred)
    acc = np.sum(y_pred_class == y_true) / dataNum
    return loss, acc


def predict_class(X, w, b):
    return np.round(forward(X, w, b)).astype(np.int)


def index_shuffle(data_num, batch_size, rand_seed=10):
    idx_list = np.arange(data_num)
    np.random.seed(rand_seed)
    np.random.shuffle(idx_list)
    iter_num = data_num // batch_size
    return np.reshape(idx_list[: iter_num * batch_size],
                      (iter_num, batch_size))


def train(
        X_Train, Y_Train, X_Val, Y_Val,
        epoch=40, batch_size=32, learning_rate=1e-2, reg_lambda=1e-3):
    # TODO: split whole dataset into train and validation part
    w = np.zeros((X_Train.shape[1]))
    b = np.zeros((1))

    train_loss = np.zeros((epoch + 1))
    train_acc = np.zeros((epoch + 1))
    train_loss[0], train_acc[0] = compute_loss_and_acc(X_Train, Y_Train, w, b, reg_lambda=reg_lambda)

    val_loss = np.zeros((epoch + 1))
    val_acc = np.zeros((epoch + 1))
    val_loss[0], val_acc[0] = compute_loss_and_acc(X_Val, Y_Val, w, b, reg_lambda = reg_lambda)

    prev_w_gra = np.zeros((X_Train.shape[1]))
    prev_b_gra = np.zeros((1))
    for e in tqdm.tqdm(range(epoch)):
        iter_idx = index_shuffle(X_Train.shape[0], batch_size)

        for i in range(iter_idx.shape[0]):
            X = X_Train[iter_idx[i]]
            Y = Y_Train[iter_idx[i]]

            w_grad, b_grad = compute_gradient(X, Y, w, b, reg_lambda = reg_lambda)

            prev_w_gra += w_grad ** 2
            prev_b_gra += b_grad ** 2

            w -= learning_rate * w_grad / np.sqrt(prev_w_gra + 1e-6)
            b -= learning_rate * b_grad / np.sqrt(prev_b_gra + 1e-6)

        train_loss[e+1], train_acc[e+1] = compute_loss_and_acc(X_Train, Y_Train, w, b, reg_lambda = reg_lambda)
        val_loss[e+1], val_acc[e+1] = compute_loss_and_acc(X_Val, Y_Val, w, b, reg_lambda = reg_lambda)

    return w, b, [train_loss, val_loss, train_acc, val_acc]


def save_logistic_model(
        w, b, feature_mean, feature_std,
        w_path, b_path, feature_mean_path, feature_std_path):
    np.save(w_path, w)
    np.save(b_path, b)
    np.save(feature_mean_path, feature_mean)
    np.save(feature_std_path, feature_std)


def load_logistic_model(w_path, b_path, feature_mean_path, feature_std_path):
    w = np.load(w_path)
    b = np.load(b_path)
    feature_mean = np.load(feature_mean_path)
    feature_std = np.load(feature_std_path)
    return w, b, feature_mean, feature_std


if __name__ == '__main__':
    # startTime = time.time()
    # Extract file path from argv
    X_train_fpath, Y_train_fpath, X_test_fpath, Pred_save_fpath, To_train = \
        Parse_argv(sys.argv)

    modelFolder = os.path.join('.', 'models', 'logistic')
    feature_m_path = os.path.join('.', "models", 'feature_mask.npz')
    logistic_w_path = os.path.join(modelFolder, 'w.npy')
    logistic_b_path = os.path.join(modelFolder, 'b.npy')
    X_train_mean_path = os.path.join(modelFolder, 'feature_mean.npy')
    X_train_std_path = os.path.join(modelFolder, 'feature_std.npy')

    if not (os.path.isfile(logistic_w_path)
            and os.path.isfile(logistic_b_path)):
        To_train = True

    if To_train:
        logistic_w_path = os.path.join(modelFolder, 'new_w.npy')
        logistic_b_path = os.path.join(modelFolder, 'new_b.npy')
        X_train_mean_path = os.path.join(modelFolder, 'new_feature_mean.npy')
        X_train_std_path = os.path.join(modelFolder, 'new_feature_std.npy')

        # Read train data
        X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
        Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)

        X_train_norm, X_train_mean, X_train_std = util.preprocessing(X_train)
        X_train_part, y_train_part, X_val_part, y_val_part = \
            util.split_train_val(X_train_norm, Y_train, val_size=0.2)

        logistic_w, logistic_b, _ = train(
                X_train_norm, Y_train, X_val_part, y_val_part,
                epoch=500, batch_size=256,
                learning_rate=1e-2, reg_lambda=1e-4)

        save_logistic_model(w = logistic_w, b = logistic_b,
                            feature_mean = X_train_mean, feature_std = X_train_std,
                            w_path = logistic_w_path, b_path = logistic_b_path,
                            feature_mean_path = X_train_mean_path, feature_std_path = X_train_std_path)
    else:
        # load model parameters
        logistic_w, logistic_b, X_train_mean, X_train_std = \
            load_logistic_model(
                    w_path=logistic_w_path,
                    b_path=logistic_b_path,
                    feature_mean_path=X_train_mean_path,
                    feature_std_path=X_train_std_path)

    feature_mask = np.load(feature_m_path)["mask"]

    # Load testing data
    X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)[:, 1:]

    # Extract features and data preprocessing
    X_test_norm = util.preprocessing(
            X_test,
            Train=False,
            X_train_mean=X_train_mean,
            X_train_std=X_train_std,
            feature_mask=feature_mask,
            add_square_and_cubic=True)

    # Predict
    Y_test_pred = predict_class(X_test_norm, logistic_w, logistic_b)

    # Save prediction result
    util.save_result(Pred_save_fpath, Y_test_pred)
    # print("Time:", round(time.time() - startTime), "seconds.")
