from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import category_encoders as ce


DATA_DIR = path.dirname(__file__)


def all_datasets():
    """
    Generator to return datasets 
    """
    datasets = [
        ("CC Defaults", load_ccdefaults),
        ("Pendigits", load_pendigits),
        ("Mushrooms", load_mushrooms())
    ]
    for ds_name, loader in datasets:
        features, labels = loader()
        features = normalize(features)

        yield ds_name, *train_test_split(features, labels, random_state=42)


def _load_file(file_name):
    data = pd.read_csv(path.join(DATA_DIR, file_name))

    labels = data.iloc[:, -1]
    features = data.iloc[:, :-1]

    return features, labels


def load_ccdefaults():
    features, labels = _load_file('ccdefaults.csv')
    return features.drop(['ID'], axis=1), labels


def load_pendigits():
    return _load_file('pendigits.csv')


def load_heartfailure():
    return _load_file('heart_failure_clinical_records_dataset.csv')


def load_mushrooms():
    data = pd.read_csv(path.join(DATA_DIR, 'agaricus-lepiota.csv'), header=None)

    labels = data.iloc[:, 0]
    labels[labels == 'e'] = 1
    labels[labels == 'p'] = 0
    labels = labels.astype(np.int8)
    enc = ce.OneHotEncoder(return_df=True, drop_invariant=True)
    features = data.iloc[:, 1:]
    enc.fit(features)
    enc_features = enc.transform(features)

    return enc_features, labels

