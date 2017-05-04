import sqlite3 as sq
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier


def ExtraTrees(table, result):
    conn = sq.connect('../database.db')
    dataset = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
    conn.close()
    dataset.drop(['index'], axis=1, inplace=True)
    dataset['HTFTR'].replace(result, 1, inplace=True)
    dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

    # Half-time/Full-time combinations are 9
    assert len(dataset['HTFTR'].unique()) == 2
    all_inputs = dataset.drop('HTFTR', axis=1).values
    all_classes = dataset['HTFTR'].values

    # Feature Importance with Extra Trees Classifier
    model = ExtraTreesClassifier()
    model.fit(all_inputs, all_classes)
    print(model.feature_importances_)

def KBest(table, result):
    conn = sq.connect('../database.db')
    dataset = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
    conn.close()
    dataset.drop(['index'], axis=1, inplace=True)
    dataset['HTFTR'].replace(result, 1, inplace=True)
    dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

    # Half-time/Full-time combinations are 9
    assert len(dataset['HTFTR'].unique()) == 2
    all_inputs = dataset.drop('HTFTR', axis=1).values
    all_classes = dataset['HTFTR'].values

    # Feature Selection

    model = SelectKBest(score_func='f_classif', k=all)
    scores = model.fit(all_inputs, all_classes)

if __name__ == '__main__':
    ExtraTrees('numpy_5', 31)
