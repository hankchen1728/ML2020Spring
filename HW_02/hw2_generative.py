import os
import sys
import numpy as np
from numpy.linalg import inv

import util


def Parse_argv(argv_list):
    To_train = False
    if len(argv_list) > 5 and argv_list[5] == "--Train":
        To_train = True
    X_train_fpath = argv_list[1]
    Y_train_fpath = argv_list[2]
    X_test_fpath = argv_list[3]
    Pred_save_fpath = argv_list[4]

    return (X_train_fpath,
            Y_train_fpath,
            X_test_fpath,
            Pred_save_fpath,
            To_train)


# generative model using functions
def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)


def forward(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def compute_acc(X, y_true, w, b):
    dataNum = y_true.shape[0]
    y_pred_class = predict_class(X, w, b)
    acc = np.sum(y_pred_class == y_true) / dataNum
    return acc


def predict_class(X, w, b):
    y_out = forward(X, w, b)
    y_class = np.ones((X.shape[0])).astype(np.int)
    y_class[y_out > 0.5] = 0
    return y_class


def Train(X, y):
    class_0_id = (y == 0)
    class_1_id = (y == 1)

    class_0 = X[class_0_id, :]
    class_1 = X[class_1_id, :]

    mean_0 = np.mean(class_0, axis=0)
    mean_1 = np.mean(class_1, axis=0)

    featureNum = class_0.shape[1]
    cov_0 = np.zeros((featureNum, featureNum))
    cov_1 = np.zeros((featureNum, featureNum))

    for i in range(class_0.shape[0]):
        cov_0 += np.dot(np.transpose([class_0[i] - mean_0]),
                        [(class_0[i] - mean_0)]) / class_0.shape[0]

    for i in range(class_1.shape[0]):
        cov_1 += np.dot(np.transpose([class_1[i] - mean_1]),
                        [(class_1[i] - mean_1)]) / class_1.shape[0]

    cov = (cov_0 * class_0.shape[0] + cov_1 * class_1.shape[0]) \
        / (class_0.shape[0] + class_1.shape[0])

    w = np.transpose(((mean_0 - mean_1)).dot(inv(cov)))
    b = (- 0.5) * (mean_0).dot(inv(cov)).dot(mean_0) + \
        0.5 * (mean_1).dot(inv(cov)).dot(mean_1) + \
        np.log(float(class_0.shape[0]) / class_1.shape[0])

    return w, b


def save_generative_model(
        w, b, feature_mean, feature_std,
        w_path, b_path, feature_mean_path, feature_std_path):
    np.save(w_path, w)
    np.save(b_path, b)
    np.save(feature_mean_path, feature_mean)
    np.save(feature_std_path, feature_std)


def load_generative_model(w_path, b_path, feature_mean_path, feature_std_path):
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

    modelFolder = os.path.join('.', 'models', 'generative')
    feature_m_path = os.path.join('.', "models", 'feature_mask.npz')
    generative_w_path = os.path.join(modelFolder, 'w.npy')
    generative_b_path = os.path.join(modelFolder, 'b.npy')
    X_train_mean_path = os.path.join(modelFolder, 'feature_mean.npy')
    X_train_std_path = os.path.join(modelFolder, 'feature_std.npy')

    if not (os.path.isfile(generative_w_path)
            and os.path.isfile(generative_b_path)):
        print("Now training")
        To_train = True

    feature_mask = np.load(feature_m_path)["mask"]

    if To_train:
        generative_w_path = os.path.join(modelFolder, 'new_w.npy')
        generative_b_path = os.path.join(modelFolder, 'new_b.npy')
        X_train_mean_path = os.path.join(modelFolder, 'new_feature_mean.npy')
        X_train_std_path = os.path.join(modelFolder, 'new_feature_std.npy')

    if To_train:
        # Read train data
        X_train = np.genfromtxt(
                X_train_fpath,
                delimiter=',',
                skip_header=1)[:, 1:]
        Y_train = np.genfromtxt(
                Y_train_fpath,
                delimiter=',',
                skip_header=1)[:, 1:]

        X_train_norm, X_train_mean, X_train_std = \
            util.preprocessing(
                    X_train,
                    feature_mask=feature_mask,
                    add_square_and_cubic=False)
        # X_train_part, y_train_part, X_val_part, y_val_part = \
        #     util.split_train_val(X_train_norm, Y_train, val_size=0.2)

        generative_w, generative_b = Train(X_train_norm, Y_train)

        save_generative_model(
                w=generative_w, b=generative_b,
                feature_mean=X_train_mean, feature_std=X_train_std,
                w_path=generative_w_path, b_path=generative_b_path,
                feature_mean_path=X_train_mean_path,
                feature_std_path=X_train_std_path
                )
    else:
        # load model parameters
        generative_w, generative_b, X_train_mean, X_train_std = \
            load_generative_model(
                    w_path=generative_w_path,
                    b_path=generative_b_path,
                    feature_mean_path=X_train_mean_path,
                    feature_std_path=X_train_std_path
                    )

    # Load testing data
    X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)[:, 1:]

    # Extract features and data preprocessing
    X_test_norm = util.preprocessing(
            X_test,
            Train=False,
            X_train_mean=X_train_mean,
            X_train_std=X_train_std,
            feature_mask=feature_mask,
            add_square_and_cubic=False)

    # Predict
    Y_test_pred = predict_class(X_test_norm, generative_w, generative_b)

    # Save prediction result
    util.save_result(Pred_save_fpath, Y_test_pred)
    # print("Time:", round(time.time() - startTime), "seconds.")
