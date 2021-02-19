from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostClassifier

from omscs_7641_p1.datasets import all_datasets


def main():
    for dataset_name, features, labels in all_datasets():
        features = normalize(features)
        train_X, test_X, train_Y, test_Y = train_test_split(features, labels, random_state=42)
        score = run_svc(train_X, train_Y, test_X, test_Y)
        print(f"{dataset_name}: {score}")


def run_svc(train_X, train_Y, test_X, test_Y, kernel='rbf'):

    clf = SVC(kernel=kernel, max_iter=5000)
    print("fitting!")
    clf.fit(train_X, train_Y)
    print("done.")
    return clf.score(test_X, test_Y)


if __name__ == '__main__':
    main()
