from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from omscs_7641_p1.datasets import all_datasets


def main():
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        params = {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['auto', 'scale']
        }
        clf = GridSearchCV(SVC(), params, scoring='accuracy')
        clf.fit(train_X, train_Y)
        test_score = clf.score(test_X, test_Y)
        train_score = clf.score(train_X, train_Y)
        print(f"{dataset_name}: {test_score} / {train_score}")


if __name__ == '__main__':
    main()