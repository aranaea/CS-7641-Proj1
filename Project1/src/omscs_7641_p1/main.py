from sklearn.neural_network import MLPClassifier

from omscs_7641_p1.datasets import all_datasets


def main():
    clf = MLPClassifier(max_iter=700)
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():

        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        print(f"{dataset_name}: {score}")


if __name__ == '__main__':
    main()