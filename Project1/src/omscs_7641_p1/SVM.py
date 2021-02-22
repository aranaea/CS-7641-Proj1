from time import time

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


def kernel_plots(train_X, train_Y, test_X, test_Y, dataset_name):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    scores = []

    clf = SVC()
    for kernel in kernels:
        start = time()
        print(kernel)
        clf.kernel = kernel
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        scores.append(score)
        print(f"  {time() - start}::{score}")

    ax = plt.subplot()
    ax.grid(zorder=0)
    ax.set_title(f"SVM Kernel {dataset_name}")
    ax.bar(kernels, scores, zorder=3)
    ax.set_ylabel("Accuracy")
    plt.show()


def degree_plots(train_X, train_Y, test_X, test_Y, dataset_name):
    scores = []
    degrees = list(set(np.linspace(3, int(len(train_X.columns)), 5, dtype=np.int16)))

    print(f"{dataset_name}.{len(degrees)}")
    clf = SVC()
    for degree in degrees:
        start = time()
        print(f"  {degree}")
        clf.degree = degree
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        scores.append(score)
        print(f"  {time() - start}::{score}")

    ax = plt.subplot()
    ax.grid(zorder=0)
    ax.set_title(f"SVM Poly Dim {dataset_name}")
    ax.plot(degrees, scores, "o-", zorder=3)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Poly Degrees")
    plt.legend(loc='best')
    plt.show()


def kernel_search(train_X, train_Y, test_X, test_Y, dataset_name):
    params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    clf = GridSearchCV(SVC(), params, scoring='accuracy')
    print(' fitting to get scores')
    clf.fit(train_X, train_Y)
    test_score = clf.score(test_X, test_Y)
    train_score = clf.score(train_X, train_Y)
    print(f"{dataset_name}::{clf.best_params_}: {test_score} / {train_score}")
    return clf.best_params_


def degree_search(train_X, train_Y, test_X, test_Y, dataset_name):
    max_dim = int(len(train_X.columns) * 0.666)
    print(f"Checking up to {max_dim} for {dataset_name}")
    params = {'degree': list(range(3, max_dim))}
    clf = GridSearchCV(SVC(kernel='poly'), params, scoring='accuracy')
    print(' fitting to get scores')
    clf.fit(train_X, train_Y)
    test_score = clf.score(test_X, test_Y)
    train_score = clf.score(train_X, train_Y)
    print(f"  {dataset_name}::{clf.best_params_}: {test_score} / {train_score}")
    return clf.best_params_


def main():
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        params = kernel_search(train_X, train_Y, test_X, test_Y, dataset_name)
        print(params)
        params = degree_search(train_X, train_Y, test_X, test_Y, dataset_name)
        print(params)
        kernel_plots(train_X, train_Y, test_X, test_Y, dataset_name)
        degree_plots(train_X, train_Y, test_X, test_Y, dataset_name)
        clf = SVC()
        if dataset_name == "Diamonds":
            clf.kernel = 'linear'
        clf.fit(train_X, train_Y)
        draw_learning_curve(clf, test_X, test_Y, dataset_name)


if __name__ == '__main__':
    main()