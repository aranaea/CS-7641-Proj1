from time import time

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


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
        start = time()
        # params = kernel_search(train_X, train_Y, test_X, test_Y, dataset_name)
        param_update = degree_search(train_X, train_Y, test_X, test_Y, dataset_name)




if __name__ == '__main__':
    main()