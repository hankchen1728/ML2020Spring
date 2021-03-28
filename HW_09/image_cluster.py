import os
import time
import argparse
import pickle

import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

from net import AutoEncoder
from data_loader import get_train_valid_dataLoader


def invert(pred):
    return np.abs(1-pred)


def cal_acc(gt: np.ndarray, pred: np.ndarray):
    """ Computes categorization accuracy of our task.

    Args:
      gt: Ground truth labels (NUM,)
      pred: Predicted labels (NUM,)
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    assert gt.shape == pred.shape, "Mismatch shape between `gt` and `pred`"
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    if acc < 0.5:
        print("Please invert the clustering prediction!!!")
    return max(acc, 1-acc)


def save_prediction(pred, out_csv="prediction.csv"):
    with open(out_csv, 'w') as f:
        f.write("id,label\n")
        for i, p in enumerate(pred):
            f.write("%d,%d\n" % (i, p))
    print("Save prediction to %s" % out_csv)


def image_encoding(image_fpath, net, device="cuda", batch_size=256):
    """Perform model inference to get encoded latent vectors"""
    img_loader = get_train_valid_dataLoader(
        npy_fpath=image_fpath,
        batch_size=batch_size,
        train_shuffle=False,
        valid_split=0)

    net.eval()
    encoded_latents = list()
    with torch.no_grad():
        for batch_idx, (data) in enumerate(img_loader):
            data = data.to(device)
            encoded, reconst = net(data)

            # storing the encoded latent
            latent = encoded.view(encoded.size()[0], -1).cpu().detach().numpy()
            encoded_latents.append(latent)

    # Concate the latent of each batch
    encoded_latents = np.concatenate(encoded_latents, axis=0)
    print("Encoded latent shape:", encoded_latents.shape)
    return encoded_latents


def clustering(
        latent_vec,
        pca_ndim=200,
        random_seed=10,
        model_folder="./checkpoints/",
        kpca_transformer=None,
        mbKMeans=None,
        store=False):
    """Perform PCA Dim reduction and then Kmeans clustering"""
    # PCA Dimension Reduction
    if kpca_transformer is None:
        transformer = KernelPCA(
            n_components=pca_ndim,
            kernel='rbf',
            n_jobs=16,
            random_state=random_seed)
        kpca_latent = transformer.fit_transform(latent_vec)
        # Save kpca model
        if store:
            with open(os.path.join(model_folder, "kpca.pkl"), "wb") as opf:
                pickle.dump(transformer, opf)
    else:
        kpca_latent = kpca_transformer.transform(latent_vec)
    print("First Reduction Shape:", kpca_latent.shape)

    # t-SNE Dimension Reduction
    transformer = TSNE(n_components=2, random_state=random_seed)
    tsne_latent = transformer.fit_transform(kpca_latent)
    print("Second Reduction Shape:", tsne_latent.shape)

    if mbKMeans is None:
        # Clustering
        kmeans = MiniBatchKMeans(
            n_clusters=2,
            max_iter=300,
            random_state=random_seed).fit(tsne_latent)
        cluster_indices = kmeans.labels_
        if store:
            with open(os.path.join(model_folder, "mbKMeans.pkl"), "wb") as opf:
                pickle.dump(kmeans, opf)
    else:
        cluster_indices = mbKMeans.predict(tsne_latent)

    return cluster_indices, kpca_latent


def load_sklearn_models(model_base_dir):
    """Load the KPCA and MiniBatchKMeans models from pickle paths

    Return "None" if the pickle file does not exist
    """
    kpca_fpath = os.path.join(model_base_dir, "kpca.pkl")
    mbKMeans_fpath = os.path.join(model_base_dir, "mbKMeans.pkl")

    kpca = None
    if os.path.isfile(kpca_fpath):
        with open(kpca_fpath, "rb") as opf:
            kpca = pickle.load(opf)
    else:
        print("`KPCA` pickle file does NOT exist!")

    mbKMeans = None
    if os.path.isfile(mbKMeans_fpath):
        with open(mbKMeans_fpath, "rb") as opf:
            mbKMeans = pickle.load(opf)
    else:
        print("`MiniBatchKMeans` pickle file does NOT exist!")

    return kpca, mbKMeans


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random_seed = args.random_seed

    # Load model weights
    net = AutoEncoder()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)

    checkpoint = torch.load(args.ckpt_path)
    print("Epoch %03d, Valid Loss: %2.6f" %
          (checkpoint["epoch"], checkpoint["val_loss"]))

    net.load_state_dict(checkpoint["net"], False)
    net.eval()

    # Predict latent
    image_fpath = args.trainX_fpath
    encoded_latent = image_encoding(image_fpath, net, device)

    # Clustering
    print("Performing dimension reduction and clustering...")
    pca_ndim = args.pca_ndim
    sk_model_base = args.sklearn_model_base
    if args.train_clustering:
        cluster_indices, _ = clustering(
            encoded_latent,
            pca_ndim=pca_ndim,
            random_seed=random_seed,
            model_folder=sk_model_base)
    else:
        kpca, mbkmeans = load_sklearn_models(sk_model_base)
        cluster_indices, _ = clustering(
            encoded_latent,
            pca_ndim=pca_ndim,
            random_seed=random_seed,
            model_folder=sk_model_base,
            kpca_transformer=kpca,
            mbKMeans=mbkmeans)

    cluster_indices = invert(cluster_indices)
    save_prediction(cluster_indices, out_csv=args.predict_csv)

    # Evaluation if providing ground truth labels
    label_fpath = args.label_fpath
    if isinstance(label_fpath, str) and os.path.isfile(label_fpath):
        ground_truth = np.load(label_fpath)
        cluster_acc = cal_acc(ground_truth, cluster_indices)
        print("Predict clustering accuracy: %2.3f" % cluster_acc)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--trainX_fpath",
            type=str,
            default="./data/trainX.npy")

    parser.add_argument(
            "--ckpt_path",
            type=str,
            default="./checkpoints/baseline.path")

    parser.add_argument(
            "--predict_csv",
            type=str,
            default="./prediction.csv")

    parser.add_argument(
            "--label_fpath",
            type=str,
            default=None)

    parser.add_argument(
            "--train_clustering",
            action="store_true",
            default=False)

    parser.add_argument(
            "--pca_ndim",
            type=int,
            default=200)

    parser.add_argument(
            "--sklearn_model_base",
            type=str,
            default="./checkpoints/")

    parser.add_argument(
            "--random_seed",
            type=int,
            default=10)

    args = parser.parse_args()

    main(args)

    print("Running time %4.1f sec(s)" % (time.time() - start_time))
