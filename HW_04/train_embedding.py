import os
# import numpy as np
# import pandas as pd
import argparse
from gensim.models import word2vec
from data_loader import load_training_data
from data_loader import load_testing_data


def train_word2vec(x, vec_size=250, num_workers=16):
    """Training the word to vector embedding model"""
    model = word2vec.Word2Vec(
        x,
        size=vec_size,
        window=5,
        min_count=5,
        workers=num_workers,
        iter=10,
        sg=1)
    return model


def main(args):
    train_fpath = os.path.join(args.data_dir, "training_label.txt")
    train_nolabel_fpath = os.path.join(args.data_dir, "training_nolabel.txt")
    test_fpath = os.path.join(args.data_dir, "testing_data.txt")

    # Loading the train and test data
    print("loading training data ...")
    train_x, y = load_training_data(train_fpath)
    train_x_no_label = load_training_data(train_nolabel_fpath)

    print("loading testing data ...")
    test_x = load_testing_data(test_fpath)

    model = train_word2vec(train_x + train_x_no_label + test_x,
                           vec_size=args.vector_size)
    # model = train_word2vec(train_x + test_x, vec_size=args.vector_size)

    print("saving model ...")
    model.save(os.path.join(args.ckpt_dir,
                            "w2v_all_{}.model".format(args.vector_size)
                            ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training word embedding")

    parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/")

    parser.add_argument(
            "--ckpt_dir",
            type=str,
            default="./checkpoint")

    parser.add_argument(
            "--vector_size",
            type=int,
            default=250)

    args = parser.parse_args()

    main(args)
