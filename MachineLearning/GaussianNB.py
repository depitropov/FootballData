import numpy as np
import pandas as pd
import sqlite3 as sq
import time


from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib
from fixtures_setup import init_fixtures


def create_estimator(table, result):
    # Convert 1/x(2/x) to 1 and all other to 0 and drop NA containing rows.
    conn = sq.connect('../database.db')
    dataset = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
    dataset.drop(['index', 'league_id', 'country_id'], axis=1, inplace=True)
    dataset['HTFTR'].replace(result, 1, inplace=True)
    dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

    # Half-time/Full-time combinations are 9

    assert len(dataset['HTFTR'].unique()) == 2

    all_inputs = dataset.drop('HTFTR', axis=1).values
    all_classes = dataset['HTFTR'].values

    # Split-out validation dataset
    validation_size = 0.20
    X_train, X_validation, Y_train, Y_validation = \
        model_selection.train_test_split(all_inputs, all_classes, test_size=validation_size)


    # grid-test
    # K-fold!!!!!!

    # Make predictions on validation dataset
    knn = GaussianNB()
    knn.fit(X_train, Y_train)
    joblib.dump(knn, '../estimators/knn_{0}_{1}.pkl'.format(table, str(result)))
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def estimate_fixtures(fixtures):
    knn = joblib.load('../estimators/knn_numpy_5_13.pkl')
    dataset = fixtures.values
    prediction = knn.predict(dataset)
    print dataset
    print prediction

if __name__ == '__main__':
    #create_estimator('numpy_5', 13)
    #create_estimator('numpy_10', 13)
    #create_estimator('numpy_5', 23)
    #create_estimator('numpy_10', 23)
    estimate_fixtures(init_fixtures.fixtures_np())
