from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from omscs_7641_p1.datasets import all_datasets


def main():
    tree = DecisionTreeClassifier(max_depth=2)
    clf = AdaBoostClassifier(tree, random_state=42, n_estimators=100)
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        print(f"{dataset_name}: {score}")


if __name__ == '__main__':
    main()