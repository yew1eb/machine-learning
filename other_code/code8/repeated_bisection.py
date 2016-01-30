#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans


def init_clusters(X):
    """
    Init cluster.

    Args:
        X: feature vector
    Returns:
        clusters: cluster<idx, samples>
        cluster_centers: cluster_centers<idx, center>
    """
    clusters = defaultdict(np.array)
    clusters[0] = X

    cluster_centers = defaultdict(np.array)
    cluster_centers[0] = np.mean(clusters[0], axis=0)

    return clusters, cluster_centers


def choose_cluster(clusters, cluster_centers):
    """
    Choose cluster based on size.

    Args:
        clusters: cluster<idx, samples>
        cluster_centers: cluster_centers<idx, centers>
    Return:
        a cluster including maximum size of samples
    """
    lens = [len(clusters[idx]) for idx in clusters]
    max_idx = np.argsort(np.array(lens))[::-1][0]
    return max_idx, clusters[max_idx], cluster_centers[max_idx]


def remove_cluster(cidx, clusters, cluster_centers):
    del clusters[cidx]
    del cluster_centers[cidx]


def choose_randomly(cluster):
    """
    choose cluster centers at random.

    Args:
        cluster: sample points
    Returns:
        cluster_centers: shape=(2, 2)
    """
    idx = np.random.randint(0, len(cluster), 2)
    cluster_centers = cluster[idx]
    return cluster_centers


def cosine_similarity(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return numerator / denominator if denominator != 0 else 0


def calc_cluster_centers(clusters):
    cluster_centers = defaultdict(np.array)
    for idx in clusters:
        center = np.mean(clusters[idx], axis=0)
        cluster_centers[idx] = center
    return cluster_centers


def bisection(cluster):
    """
    Find two sub-clusters using kmeans algorithm.
    (bisecting step)

    Args:
        cluster:
    Returns:
        bisec_clusters:
    """
    km = KMeans(n_clusters=2)
    km.fit(cluster)
    pred_labels = km.predict(cluster)

    bisec_clusters = defaultdict(list)
    for label in np.unique(pred_labels):
        bisec_clusters[label] = cluster[np.where(pred_labels == label)]

    return bisec_clusters, km.cluster_centers_


def append_cluster(max_bisec_cluster, max_bisec_cluster_centers,
                   clusters, cluster_centers):
    """
    Append bisectioned cluster into clusters.
    """
    next_id = np.max(clusters.keys())+1 if len(clusters) != 0 else 0
    for i, id in enumerate(range(next_id, next_id+2)):
        clusters[id] = max_bisec_cluster[i]
        cluster_centers[id] = max_bisec_cluster_centers[i]


def overall_similarity(bisec_clusters):
    overall_sim = 0.0
    for cidx, cluster in bisec_clusters.items():
        for sample in cluster:
            overall_sim += np.dot(sample, sample)
    return overall_sim


def repeated_bisection(X, n_clusters, ITER=100):
    # initial cluster contains all samples
    clusters, cluster_centers = init_clusters(X)

    while len(clusters) != n_clusters:
        # choose cluster to split
        cidx, cluster, cluster_center = choose_cluster(clusters,
                                                       cluster_centers)
        # remove chosen cluster from a list
        remove_cluster(cidx, clusters, cluster_centers)

        # do bisecting kmeans
        max_sim = float("-inf")
        max_bisec_clusters = None
        max_bisec_cluster_centers = None
        for iter in range(ITER):
            # bisecting chosen cluster
            bisec_clusters, bisec_cluster_centers = bisection(cluster)

            # save bisec_clusters with the highest overall simlarity
            overall_sim = overall_similarity(bisec_clusters)
            if max_sim < overall_sim:
                max_bisec_clusters = bisec_clusters
                max_sim = overall_sim
                max_bisec_cluster_centers = bisec_cluster_centers

        # append bisec_cluster to clusters
        append_cluster(max_bisec_clusters, max_bisec_cluster_centers,
                       clusters, cluster_centers)

    return clusters, cluster_centers


def show_results(clusters, cluster_centers, n_clusters):
    print '** Results'
    print 'n_clusters:', len(clusters)
    print 'centers:'
    for i in range(n_clusters):
        print cluster_centers[i]


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # do repeated bisection clustering
    n_clusters = 2
    clusters, cluster_centers = repeated_bisection(X, n_clusters)

    # show results
    show_results(clusters, cluster_centers, n_clusters)

if __name__ == '__main__':
    main()
