from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


def main():

    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        print(f"starting {dataset_name}")
        for n in range(100, 1001, 100):
            start = time()
            max_depth = 12 if dataset_name == 'Diamonds' else 5
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            clf = AdaBoostClassifier(tree, random_state=42, n_estimators=n, learning_rate=1.75)
            # draw_learning_curve(clf, train_X, train_Y, f"Boosting {dataset_name}")
            clf.fit(train_X, train_Y)
            test_score = clf.score(test_X, test_Y)
            train_score = clf.score(train_X, train_Y)
            print(f"{dataset_name} {n} - {test_score}/{train_score} :: {time() - start}s.")


if __name__ == '__main__':
    main()
