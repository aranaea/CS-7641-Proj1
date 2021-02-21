"""
Extra things to consider here:
  https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
  Change the hidden layers?
  Solvers
"""
from time import time
from collections import defaultdict

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np

from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


def build_complexity_graph(train_X, test_X, train_Y, test_Y, dataset_name, params=None):
    accuracy_scores = []

    params = params or {}
    params['activation'] = params.get('activation') or 'logistic'
    params['max_iter'] = params.get('max_iter') or 50
    params['early_stopping'] = False
    params['random_state'] = 42
    params['warm_start'] = True

    clf = MLPClassifier(**params)

    for epochs in range(20):
        start = time()

        clf.fit(train_X, train_Y)
        accuracy_scores.append((epochs*100, clf.score(test_X, test_Y), clf.score(train_X, train_Y)))
        print(f' {time() - start} - {epochs} - {clf.score(test_X, test_Y)}')

    scores = np.array(accuracy_scores)
    test_scores = scores[:, 1]
    train_scores = scores[:, 2]

    axes = plt.subplot()
    axes.set_title(f"ANN Complexity {dataset_name}")
    axes.set_xlabel("Epochs")
    axes.set_ylabel("Accuracy")
    axes.plot(scores[:, 0], train_scores, 'o-', color='r', label="Training")
    axes.plot(scores[:, 0], test_scores, 'o-', color='g', label="Test")
    axes.legend(loc="best")

    plt.show()


def explore_learning_rate(train_X, test_X, train_Y, test_Y, dataset_name, params=None):
    accuracy_scores = []

    params = params or {}
    params['activation'] = params.get('activation') or 'logistic'
    params['max_iter'] = params.get('max_iter') or 50
    params['random_state'] = 42

    clf = MLPClassifier(**params)

    for lr_init in np.linspace(0.0001, 0.0015, 10):
        start = time()
        clf.learning_rate_init = lr_init
        print(f" fitting {dataset_name} with {lr_init}")
        clf.fit(train_X, train_Y)
        accuracy_scores.append((lr_init, clf.score(test_X, test_Y), clf.score(train_X, train_Y)))
        print(f' {time() - start} - :{lr_init} - {clf.score(test_X, test_Y)}')

    axes = plt.subplot()
    axes.set_title(f"ANN Learning Rates {dataset_name}")
    axes.set_xlabel("Learning Rate")
    axes.set_ylabel("Accuracy")

    scores = np.array(accuracy_scores)
    test_scores = scores[:, 1]
    train_scores = scores[:, 2]

    axes.plot(scores[:, 0], train_scores, 'o-', color='r', label="Training")
    axes.plot(scores[:, 0], test_scores, 'o-', color='g', label="Test")

    axes.legend(loc="best")
    plt.show()


def explore_hidden_layers(train_X, test_X, train_Y, test_Y, dataset_name, params=None):
    accuracy_scores = []

    feature_count = len(train_X.columns)

    params = params or {}
    params['activation'] = params.get('activation') or 'logistic'
    params['max_iter'] = params.get('max_iter') or 50
    params['random_state'] = 42

    clf = MLPClassifier(**params)

    for layer_size in np.linspace(feature_count//2, 200, 5, dtype=np.int16):
        start = time()
        clf.hidden_layer_sizes = (layer_size, )
        print(f" fitting {dataset_name} with {layer_size}")
        clf.fit(train_X, train_Y)
        accuracy_scores.append((layer_size, clf.score(test_X, test_Y), clf.score(train_X, train_Y)))
        print(f' {time() - start} - :{layer_size} - {clf.score(test_X, test_Y)}')

    axes = plt.subplot()
    axes.set_title(f"ANN Learning Rates {dataset_name}")
    axes.set_xlabel("Hidden Layer Size")
    axes.set_ylabel("Accuracy")

    scores = np.array(accuracy_scores)
    test_scores = scores[:, 1]
    train_scores = scores[:, 2]

    axes.plot(scores[:, 0], train_scores, 'o-', color='r', label="Training")
    axes.plot(scores[:, 0], test_scores, 'o-', color='g', label="Test")

    axes.legend(loc="best")
    plt.show()


def run_grid_search():

    # params = {'activation': ['logistic', 'tanh', 'relu'],
    #           'max_iter': [700]}
    # if dataset_name == 'Pendigits':
    #     hidden_layer_sizes=(230,)
    #     params['hidden_layer_sizes'] = [(180,), (200,), (230,), (260, ), (300,)]
    # else:
    #     params['hidden_layer_sizes'] = [(13,), (14,), (15,), (16,), (17,), (18,), (19,)]
    #
    # clf = GridSearchCV(MLPClassifier(), params, scoring='accuracy')
    # params = clf.best_params_
    pass


def main():

    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        start = time()
        print(f"starting on {dataset_name}")

        if dataset_name == 'Pendigits':
            # build_complexity_graph(train_X, test_X, train_Y, test_Y, dataset_name, {'max_iter': 10})
            # explore_learning_rate(train_X, test_X, train_Y, test_Y, dataset_name, {'max_iter': 400})
            # explore_hidden_layers(train_X, test_X, train_Y, test_Y, dataset_name, {'max_iter': 400})
            draw_learning_curve(MLPClassifier(max_iter=400, activation='logistic'),
                                test_X, test_Y, f"ANN {dataset_name}")
        else:
            # build_complexity_graph(train_X, test_X, train_Y, test_Y, dataset_name)
            # explore_learning_rate(train_X, test_X, train_Y, test_Y, dataset_name, {'max_iter': 1100})
            # explore_hidden_layers(train_X, test_X, train_Y, test_Y, dataset_name, {'max_iter': 1100})
            draw_learning_curve(MLPClassifier(max_iter=800, activation='logistic'),
                                test_X, test_Y, f"ANN {dataset_name}")



if __name__ == '__main__':
    main()