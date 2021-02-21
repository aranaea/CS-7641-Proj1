"""
Extra things to consider here:
  https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
  Change the hidden layers?
  Solvers
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


def main():

    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        if dataset_name != 'Pendigits':
            continue

        print(f"starting on {dataset_name}")
        # params = {'activation': ['logistic', 'tanh', 'relu'],
        #           'max_iter': [700]}
        # if dataset_name == 'Pendigits':
        #     params['hidden_layer_sizes'] = [(180,), (200,), (230,), (260, ), (300,)]
        # else:
        #     params['hidden_layer_sizes'] = [(13,), (14,), (15,), (16,), (17,), (18,), (19,)]
        #
        # clf = GridSearchCV(MLPClassifier(), params, scoring='accuracy')

        clf = MLPClassifier(**{'activation': 'logistic', 'hidden_layer_sizes': (230,), 'max_iter': 1000})
        clf.fit(train_X, train_Y)
        # print(clf.best_params_)
        score = clf.score(test_X, test_Y)
        print(f"{dataset_name}: {score}")

        draw_learning_curve(clf, test_X, test_Y, dataset_name)


if __name__ == '__main__':
    main()