import numpy as np
import pandas as pd
import sqlite3 as sq
from time import time


from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import preprocessing
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def parameter_tune(table, result):
    # Report for parameter tuner
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
    # Load Data
    conn = sq.connect('../database.db')
    dataset = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
    #dataset.drop(['index', 'league_id', 'country_id'], axis=1, inplace=True)
    dataset['HTFTR'].replace(result, 1, inplace=True)
    dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

    # Half-time/Full-time combinations are 9

    assert len(dataset['HTFTR'].unique()) == 2

    #print dataset.columns
    dataset = dataset.loc[dataset['league_id'] == 10]
    dataset = dataset[['HTFTR',
                       'home_first_half_goals', 'home_second_half_goals',
                       'away_first_half_goals', 'away_second_half_goals',
                       'away_win_odds', 'away_lose_odds',
                       'home_win_odds', 'home_lose_odds',
                       'home_half_wins', 'home_half_draws',
                       'away_half_lose', 'away_half_draws',
                       'home_full_lose', 'home_full_draws',
                       'away_full_wins', 'away_full_draws',
                       'B365H', 'B365D', 'B365A',
                       ]]
    all_inputs = dataset.drop('HTFTR', axis=1).values
    all_inputs = preprocessing.scale(all_inputs)
    all_classes = dataset['HTFTR'].values

    # Split-out validation dataset
    validation_size = 0.20
    X_train, X_validation, Y_train, Y_validation = \
        model_selection.train_test_split(all_inputs, all_classes, test_size=0.2)
    print X_train.head()
    # Find best HyperParameters
    clf = SGDClassifier()
    #pca = PCA()
    #pca.fit(X_train)
    #X_t_train = pca.transform(X_train)
    #X_t_test = pca.transform(X_validation)
    param_grid = {"loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_epsilon_insensitive'],
                  "n_iter": [1, 3, 5, 7, 9, 10, 13, 15, 17, 20],
                  "alpha": [0.0001, 0.0003, 0.0004, 0.0005, 0.001, 0.01, 0.1, 1, 10, 100],
                  "l1_ratio": [0.00, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                  "fit_intercept": [True, False],
                  #"learning_rate": ['constant', 'optimal', 'invscaling'],
                  "class_weight": [{0: 0.94, 1: 0.06}],
                  "warm_start": [True, False],
                  "penalty": ['none', 'l2', 'l1', 'elasticnet']}

    param_dist = {"n_iter": randint(1, 11),
                  "alpha": uniform(scale=0.01),
                  "penalty": ["none", "l1", "l2"]}
    n_iter_search = 50
    grid_search = RandomizedSearchCV(clf, param_distributions=param_dist, scoring='precision', n_jobs=7)
    start = time()
    grid_search.fit(X_train, Y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_)))
    print report(grid_search.cv_results_)


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
    knn = SGDClassifier()
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
    #estimate_fixtures(init_fixtures.fixtures_np())
    parameter_tune('numpy_5', 31)
    parameter_tune('numpy_10', 31)
