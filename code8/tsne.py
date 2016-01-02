#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# from http://alexanderfabisch.github.io/t-sne-in-scikit-learn.html
#
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import random


def plot_mnist(X, y, X_embedded, name, min_dist=10.0):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title("\\textbf{MNIST dataset} -- Two dimensional "
              "embedding of 70,000 handwritten digits with %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                        wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                c=y, marker="x")

    if min_dist is not None:
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=plt.cm.gray_r), X_embedded[i])
            ax.add_artist(imagebox)

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', data_home='.')
    X, y = mnist.data / 255.0, mnist.target
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    n_train_samples = 5000
    X_pca = PCA(n_components=50).fit_transform(X)
    X_train = X_pca[indices[:n_train_samples]]
    y_train = y[indices[:n_train_samples]]

    X_train_embedded = TSNE(n_components=2,
                            perplexity=40, verbose=2).fit_transform(X_train)
    plot_mnist(X[indices[:n_train_samples]], y_train, X_train_embedded,
               "t-SNE", min_dist=20.0)
    plt.show()
