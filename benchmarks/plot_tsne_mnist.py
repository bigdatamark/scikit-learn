# [MODULE DOC]
# Command-line script plotting a 2D t-SNE embedding saved by bench_tsne_mnist,
# coloured by digit label. Options: --labels (.npy of labels, default 10000-sample
# file), --embedding (.npy of shape (n, 2)). One scatter series per unique label.
# Side effects: reads files, opens a matplotlib window.
# KNOWN ISSUE: np.load is called without allow_pickle, but the labels written by
# bench_tsne_mnist are an object array, so loading them raises ValueError.
import argparse
import os.path as op

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "mnist_tsne_output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot benchmark results for t-SNE")
    parser.add_argument(
        "--labels",
        type=str,
        default=op.join(LOG_DIR, "mnist_original_labels_10000.npy"),
        help="1D integer numpy array for labels",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=op.join(LOG_DIR, "mnist_sklearn_TSNE_10000.npy"),
        help="2D float numpy array for embedded data",
    )
    args = parser.parse_args()

    X = np.load(args.embedding)
    y = np.load(args.labels)

    for i in np.unique(y):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.2, label=int(i))
    plt.legend(loc="best")
    plt.show()
