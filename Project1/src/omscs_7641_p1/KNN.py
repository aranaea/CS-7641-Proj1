import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

from omscs_7641_p1.datasets import all_datasets


def main():
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():

        k = len(train_Y.value_counts())

        for weight_type in ['distance', 'uniform']:
            clf = KNeighborsClassifier(n_neighbors=k, weights=weight_type)
            clf.fit(train_X, train_Y)

            score = clf.score(test_X, test_Y)
            print(f"{dataset_name}:{weight_type} {score}")

            plot_confusion_matrix(clf, test_X, test_Y)
            plt.show()


if __name__ == '__main__':
    main()