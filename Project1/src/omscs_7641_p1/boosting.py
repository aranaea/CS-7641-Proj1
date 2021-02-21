from time import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

from omscs_7641_p1.datasets import all_datasets
from omscs_7641_p1.plots import draw_learning_curve


def main():

    for dataset_name, train_X, test_X, train_Y, test_Y in all_datasets():
        print(f"starting {dataset_name}")
        scores = defaultdict(list)
        for l in np.linspace(0.75, 2.0, 3):
            for n in range(100, 1101, 250):
                start = time()
                max_depth = 12 if dataset_name == 'Diamonds' else 5
                max_depth = 1
                tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                clf = AdaBoostClassifier(tree, random_state=42, n_estimators=n, learning_rate=l)
                clf.fit(train_X, train_Y)

                test_score = clf.score(test_X, test_Y)
                train_score = clf.score(train_X, train_Y)
                scores[l].append((n, test_score, train_score))
                print(f"{dataset_name} {n} - {test_score}/{train_score} :: {time() - start}s.")
        draw_plot(scores, dataset_name)


def draw_plot(scores, dataset_name):
    axes = plt.subplot()
    axes.set_title(f"Boosting {dataset_name}")
    axes.grid()

    for lr, score_list in scores.items():
        score_ary = np.array(score_list)
        axes.plot(score_ary[:, 0], score_ary[:, 1], 'o-', label=f"lr-{lr}-test")
        axes.plot(score_ary[:, 0], score_ary[:, 2], 'o-', label=f"lr-{lr}-train")

    axes.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
