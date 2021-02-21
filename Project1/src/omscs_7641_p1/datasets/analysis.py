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

def duplicates(features, labels):
    """
    For each label
    :param features:
    :param labels:
    :return:
    """



def draw_histogram(labels, title):
    bins = len(labels.value_counts())
    plt.title(title)
    plt.hist(np.array(labels), bins, density=False, facecolor='g', alpha=0.75, rwidth=0.9)
    plt.show()


def main():
    for dataset_name, load_dataset in [('Pendigits', load_pendigits),
                                       ('Diamonds', load_diamonds)]:
        features, labels = load_dataset()
        label_counts = labels.value_counts()
        draw_histogram(labels, dataset_name)
        print(dataset_name)
        print(label_counts)
        print(label_counts.mean(), label_counts.std())


if __name__ == '__main__':
    main()
