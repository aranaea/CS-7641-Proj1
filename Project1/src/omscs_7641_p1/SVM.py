from sklearn.svm import SVC
from omscs_7641_p1.datasets import all_datasets


def main():
    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():

        for kernel_type in ['linear', 'poly', 'rbf', 'sigmoid']:
            clf = SVC(kernel=kernel_type)
            clf.fit(train_X, train_Y)
            score = clf.score(test_X, test_Y)
            print(f"{dataset_name}:{kernel_type} {score}")


if __name__ == '__main__':
    main()