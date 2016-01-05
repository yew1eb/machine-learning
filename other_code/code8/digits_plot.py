#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    digits_plot [--mds]
    digits_plot -h | --help
Options:
    --mds      Use Multi Dimensional Scaling
    -h --help  show help message
"""
from docopt import docopt
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'maroon', 'indigo']


def main():
    args = docopt(__doc__)
    is_mds = args['--mds']

    # load datasets
    digits = load_digits()
    X = digits.data
    y = digits.target
    labels = digits.target_names

    # dimension reduction
    if is_mds:
        model = MDS(n_components=2)
    else:
        model = PCA(n_components=2)
    X_fit = model.fit_transform(X)

    for i in range(labels.shape[0]):
        plt.scatter(X_fit[y == i, 0], X_fit[y == i, 1],
                    color=COLORS[i], label=str(i))

    plt.legend(loc='upper left')
    plt.autoscale()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
