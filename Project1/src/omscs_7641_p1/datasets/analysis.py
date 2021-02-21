"""
Functions to provide an analysis of the selected datasets.  These are meant to provide some insights into whether
the datasets will be "interesting" for teh problems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import completeness_score, homogeneity_score

from omscs_7641_p1.datasets import load_pendigits, load_diamonds


def categorical_distance(features, labels):
    """
    If the features are grouped using the ground truth labels, how much do the
    clusters overlap?  If we check the distances between vectors?  
    :param features:
    :param labels:
    :return:
    """
    pass

def draw_all_features(features, labels, dataset_name):
    """
      Create a single plot with len(features.columns) subplots.  Each would be a scatter graph with all of the values
    """
    plot_count = len(features.columns)
    col_count = 2
    row_count = plot_count // col_count
    if plot_count == 9:
        fig, ax = plt.subplots(3, 3, gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
        col_count = 3
        row_count = 3
    elif plot_count == 16:
        fig, ax = plt.subplots(4, 4, gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
        col_count = 4
        row_count = 4
    else:
        print(f"Unexpected plot count {plot_count}")
        fig, ax = plt.subplots(plot_count//2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.3})

    fig.suptitle(f'{dataset_name.capitalize()} Feature Relation to Categories')
    for index, column in enumerate(features.columns):
        row_col = np.unravel_index([index], (row_count, col_count))
        row = row_col[0][0]
        col = row_col[1][0]
        ax[row, col].scatter(features[column], labels, alpha=0.005)
        ax[row, col].set_title(column.capitalize())
    plt.show()


def draw_histogram(labels, title):
    bins = len(labels.value_counts())
    plt.title(title)
    plt.hist(np.array(labels), bins, density=False, facecolor='g', alpha=0.75, rwidth=0.9)
    plt.show()


def main():
    for dataset_name, load_dataset in [('Pendigits', load_pendigits),
                                       ('Diamonds', load_diamonds)]:
        features, labels = load_dataset()
        draw_all_features(features, labels, dataset_name)
        # label_counts = labels.value_counts()
        # draw_histogram(labels, dataset_name)
        # print(dataset_name)
        # print(label_counts)
        # print(label_counts.mean(), label_counts.std())


if __name__ == '__main__':
    main()
