import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve


def draw_learning_curve(clf, test_X, test_Y, title='Tree', folds=10):
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf, test_X, test_Y,
                                                                          train_sizes=np.linspace(0.1, 1., folds),
                                                                          cv=folds, n_jobs=3, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes = plt.subplot()
    axes.set_title(f"Learning Curve {title}")
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")

    plt.show()