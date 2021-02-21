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
        ("Diamonds", load_diamonds),
        # ("CC Defaults", load_ccdefaults),
        ("Pendigits", load_pendigits)
    ]
    for ds_name, loader in datasets:
        features, labels = loader()

        yield ds_name, *train_test_split(features, labels, random_state=42)


def _load_file(file_name):
    data = pd.read_csv(path.join(DATA_DIR, file_name))

    labels = data.iloc[:, -1]
    features = data.iloc[:, :-1]

    return features, labels


def load_clean_ccdefaults():
    data = pd.read_csv(path.join(DATA_DIR, 'ccdefaults.csv'))
    data['ON_TIME'] = data[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].sum(axis=1)
    data['PAY_TOTAL'] = data[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)
    data['BILL_TOTAL'] = data[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]\
        .sum(axis=1)
    data['PAY_RATIO'] = data['PAY_TOTAL'] / data['BILL_TOTAL']
    data['BALANCE_RATIO'] = data['BILL_TOTAL'] / data['LIMIT_BAL']
    data = data.dropna(axis=1)
    labels = data['default payment next month']
    features = data.drop(['ID', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], axis=1)
    return features, labels


def load_ccdefaults():
    features, labels = _load_file('ccdefaults.csv')
    # express payments as a ratio of LIMIT_BAL
    return features.drop(['ID'], axis=1), labels


def load_pendigits():
    return _load_file('pendigits.csv')


def load_diamonds():
    features, labels =  _load_file('processed_diamonds.csv')
    return features.drop(['id','price'], axis=1), labels


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

