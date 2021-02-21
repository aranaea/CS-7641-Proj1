import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
import pandas as pd

from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


def cut_diamonds(features: pd.DataFrame):
    """
    Remove some features that are probably adding noise, maybe.
    """
    return features.drop(['cut', 'table', 'depth'], axis=1)


def main():
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        print(dataset_name)
        # metric = 'euclidean'
        # if dataset_name == 'Diamonds':
        #     metric = 'canberra'
        #     test_X = cut_diamonds(test_X)
        #     train_X = cut_diamonds(train_X)
        # else:
        #     test_X = test_X.drop(['X5'], axis=1)
        #     train_X = train_X.drop(['X5'], axis=1)

        k = len(train_Y.value_counts())
        clf = KNeighborsClassifier(n_neighbors=k, weights='distance')  # , metric=metric)
        clf.fit(train_X, train_Y)

        score = clf.score(test_X, test_Y)
        print(f"k.{k}:auto:{dataset_name}:{score}")

        draw_learning_curve(clf, test_X, test_Y, f"KNN {dataset_name}")
        # plot_confusion_matrix(clf, test_X, test_Y, include_values=False)
        # plt.show()


if __name__ == '__main__':
    main()