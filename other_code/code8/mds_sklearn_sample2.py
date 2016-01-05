#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib._png import read_png
import os
import re


def main():
    # load sample data
    data = np.loadtxt('distmat799.txt', delimiter=',')
    dists = data / np.amax(data)

    # load images
    img_files = [img for img in os.listdir('799_patch') if re.search(r'\.png', img)]

    # mds
    mds = MDS(n_components=2, dissimilarity='precomputed')
    results = mds.fit(dists)

    # plot
    fig, ax = plt.subplots()
    for i, img_file in enumerate(img_files):
        img_file = os.path.join('799_patch', img_file)
        img = read_png(img_file)
        imagebox = OffsetImage(img, zoom=2.0)
        coords = results.embedding_[i, :]
        xy = tuple(coords)
        ab = AnnotationBbox(imagebox, xy)
        ax.add_artist(ab)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    plt.show()

if __name__ == '__main__':
    main()
