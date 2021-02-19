from typing import Tuple
from random import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import normalize

from omscs_7641_p1.datasets import load_pendigits, load_ccdefaults, load_mushrooms, load_heartfailure


def grid_search(train_X, train_Y):
    tuning_params = [{'criterion': ['gini', 'entropy'],
                      'max_depth': np.linspace(2, 32, 10, dtype=int),
                      'random_state': [42],
                      'min_samples_split': np.linspace(2, 100, 50, dtype=int)}]
    scoring = 'accuracy'

    clf = GridSearchCV(DecisionTreeClassifier(), tuning_params, scoring=scoring)
    clf.fit(train_X, train_Y)
    return clf.best_params_


def model_complexity_graph(train_X, train_Y, params, title='depth study'):

    scores = []
    for depth in np.linspace(2, 32, 10, dtype=int):
        params['max_depth'] = depth
        clf = DecisionTreeClassifier(**params)

        train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, n_jobs=3)
        scores.append([depth, np.mean(train_scores), np.std(train_scores), np.mean(test_scores),
                       np.std(test_scores)])

    scores = np.array(scores)

    x_axis = scores[:, 0]
    axes = plt.subplot()
    axes.set_xlabel('Max Tree Depth')
    axes.set_ylabel('Accuracy')
    axes.set_title(f"Model Comlpexity {title}")
    axes.grid()
    axes.fill_between(x_axis, scores[:, 1] - scores[:, 2], scores[:, 1] + scores[:, 2], alpha=0.1, color="r")
    axes.fill_between(x_axis, scores[:, 3] - scores[:, 4], scores[:, 3] + scores[:, 4], alpha=0.1, color="g")
    axes.plot(x_axis, scores[:, 1], 'o-', color="r", label="Training Score")
    axes.plot(x_axis, scores[:, 3], 'o-', color="g", label="Validation Score")
    axes.legend(loc="best")
    plt.show()


def pruning_check(features, labels, params, title='depth study'):

    scores = []

    base_clf = DecisionTreeClassifier(**params)
    train_X, test_X, train_Y, test_Y = train_test_split(features, labels, random_state=42)

    path = base_clf.cost_complexity_pruning_path(features, labels)

    alphas = sorted(path.ccp_alphas, reverse=True)
    print(len(alphas))
    for ccp_alpha in alphas:
        print('.', end='')
        # print(f'processing {ccp_alpha}')
        params['ccp_alpha'] = ccp_alpha
        clf = DecisionTreeClassifier(**params)

        train_sizes, train_scores, test_scores = learning_curve(clf, features, labels, cv=10, n_jobs=3)
        # clf.fit(train_X, train_Y)
        # node_count = clf.tree_.node_count
        # print(node_count)
        scores.append([ccp_alpha, np.mean(train_scores), np.std(train_scores), np.mean(test_scores),
                       np.std(test_scores)])
    print('')

    scores = np.array(scores)

    x_axis = scores[:, 0]
    axes = plt.subplot()
    axes.set_xlabel('Alpha Prune')
    axes.set_ylabel('Accuracy')
    axes.set_title(f"Pruning Alpha {title}")
    axes.grid()
    axes.fill_between(x_axis, scores[:, 1] - scores[:, 2], scores[:, 1] + scores[:, 2], alpha=0.1, color="r")
    axes.fill_between(x_axis, scores[:, 3] - scores[:, 4], scores[:, 3] + scores[:, 4], alpha=0.1, color="g")
    axes.plot(x_axis, scores[:, 1], 'o-', color="r", label="Training Score")
    axes.plot(x_axis, scores[:, 3], 'o-', color="g", label="Validation Score")
    axes.legend(loc="best")
    plt.show()

    return scores[scores[:, 3].argsort()][-1][0]


def larger_tree_with_pruning(features, labels, params, dataset_name):
    """
    Test if it's better to build a bigger tree then prune vs setting a more
    restrictive max_depth
    """

    base_depth = params['max_depth']
    temp_params = params.copy()
    depths = [base_depth * 3, base_depth, base_depth // 2]
    for depth in depths:
        print(f'depth: {depth}')
        temp_params['max_depth'] = depth
        pruning_check(features, labels, temp_params, f"{dataset_name} at {depth}")


def draw_learning_curve(clf, test_X, test_Y, title='Tree'):
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf, test_X, test_Y,
                                                                          cv=10, n_jobs=3, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes = plt.subplot()
    axes.set_title(f"Learning Curve {title}")
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")

    plt.show()


def create_balance(features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vc = labels.value_counts()
    sample_count = min(vc)

    dataset = features.copy()
    dataset['labels'] = labels

    pos = dataset[labels == 1].sample(sample_count, random_state=42)
    neg = dataset[labels == 0].sample(sample_count, random_state=42)

    new_ds = pd.concat([pos, neg])
    new_ds = new_ds.reindex(np.random.permutation(new_ds.index))

    return new_ds.iloc[:, :-1], new_ds.iloc[:, -1]


def main():

    for dataset_name, load_dataset in [
                                       ('Pendigits', load_pendigits),
                                       ('CC Defaults Norm', load_ccdefaults)
                                       ]:

        print(f'Processing {dataset_name}')
        features, labels = load_dataset()

        train_X, test_X, train_Y, test_Y = train_test_split(features, labels, random_state=42)

        print("Initial score: ", end='')
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(train_X, train_Y)
        plot_tree(clf)
        plt.show()
        plot_confusion_matrix(clf, test_X, test_Y)
        plt.show()
        predictions = clf.predict(test_X)
        print(accuracy_score(test_Y, predictions))

        params = grid_search(train_X, train_Y)
        params['random_state'] = 42
        print(params)
        print("After GS score: ", end='')
        clf = DecisionTreeClassifier(**params)
        clf.fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        plot_tree(clf)
        plt.show()
        print(accuracy_score(test_Y, predictions))

        # if dataset_name == "CC Defaults":
        #     features = normalize(features)

        params['ccp_alpha'] = pruning_check(train_X, train_Y, params.copy(), dataset_name)

        print("After prune score: ", end='')
        clf = DecisionTreeClassifier(**params)
        clf.fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        plot_tree(clf)
        plt.show()
        print(accuracy_score(test_Y, predictions))

        model_complexity_graph(train_X, train_Y, params.copy(), dataset_name)

        print(f"Best we could find for {dataset_name} is {params}")
        clf = DecisionTreeClassifier(**params)
        clf.fit(train_X, train_Y)
        plot_tree(clf)
        plt.show()
        plot_confusion_matrix(clf, test_X, test_Y)
        plt.show()
        #
        draw_learning_curve(clf, test_X, test_Y, dataset_name)

        predictions = clf.predict(test_X)
        print(accuracy_score(test_Y, predictions))
        try:
            plot_roc_curve(clf, test_X, test_Y)
            plt.show()
        except ValueError:
            pass
        print('')


if __name__ == '__main__':
    main()
