from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sqlite3 as sq
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def evaluate_models(table, result):
    conn = sq.connect('../database.db')
    dataset = pd.read_sql_query('SELECT * FROM {0}'.format(table), conn)
    conn.close()
    dataset.drop(['index'], axis=1, inplace=True)
    dataset['HTFTR'].replace(result, 1, inplace=True)
    dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

    # Half-time/Full-time combinations are 9
    assert len(dataset['HTFTR'].unique()) == 2

    all_inputs = dataset.drop('HTFTR', axis=1).values
    #all_inputs = dataset[['home_first_half_goals', 'away_first_half_goals', 'away_win_odds', 'away_draw_odds', 'home_second_half_goals', 'home_lose_odds', 'home_draw_odds', 'away_lose_odds_all', 'away_second_half_goals_all', 'home_first_half_goals_all', 'home_win_odds_all', 'away_second_half_goals', 'home_win_odds', 'home_second_half_goals_all', 'home_half_lose_all', 'away_lose_odds', 'B365D', 'away_half_lose_all', 'B365A', 'home_full_lose_all', 'away_full_draws_all', 'home_lose_odds_all', 'away_home_draw_all', 'B365H', 'away_half_draws_all', 'away_half_wins_all', 'away_first_half_goals_all', 'home_home_draw_all', 'home_draw_odds_all']].values
    all_classes = dataset['HTFTR'].values
    #all_inputs = preprocessing.normalize(all_inputs)
    all_inputs = preprocessing.scale(all_inputs)

    # Split-out validation dataset
    validation_size = 0.20
    X_train, X_validation, Y_train, Y_validation = \
        model_selection.train_test_split(all_inputs, all_classes, test_size=validation_size)

    # Spot Check Algorithms
    models = [#('LR', LogisticRegression()),
              #('LDA', LinearDiscriminantAnalysis()),
              #('KNN', KNeighborsClassifier()),
              #('CART', DecisionTreeClassifier()),
              #('NB', GaussianNB()),
              #('SVM', SVC()),
              ('SGD', SGDClassifier(warm_start= True, n_iter= 8, loss= 'squared_hinge', l1_ratio= 0.5,
                                    fit_intercept= True, penalty= 'l1', alpha= 0.0003, class_weight= {0: 0.94, 1: 0.06}))
              ]
    seed = 7
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=6, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='precision')
        results.append(cv_results)
        names.append(name)
        print cv_results
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


if __name__ == '__main__':
    #conn = sq.connect('../database.db')
    #dataset = pd.read_sql_query('SELECT * FROM {0}'.format('numpy_5'), conn)
    #print dataset
    evaluate_models('numpy_5', 31)
    #evaluate_models('numpy_10', 13)
