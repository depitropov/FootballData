import numpy as np
import pandas as pd
import sqlite3 as sq
from time import time

import xgboost as xgb
from sklearn import cross_validation, metrics, model_selection
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from matplotlib.pylab import rcParams
import matplotlib.pylab as plt

rcParams['figure.figsize'] = 12, 4
#DATA

conn = sq.connect('../database.db')
dataset = pd.read_sql_query('SELECT * FROM numpy_5', conn)
dataset['HTFTR'].replace(31, 1, inplace=True)
dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

# Half-time/Full-time combinations are 9

assert len(dataset['HTFTR'].unique()) == 2

# Fix Data
classes = dataset['HTFTR'].values
dtrain = dataset[['HTFTR',
                   'home_first_half_goals', 'home_second_half_goals',
                   'away_first_half_goals', 'away_second_half_goals',
                   'away_win_odds', 'away_lose_odds',
                   'home_win_odds', 'home_lose_odds',
                   'home_half_wins', 'home_half_draws',
                   'away_half_lose', 'away_half_draws',
                   'home_full_lose', 'home_full_draws',
                   'away_full_wins', 'away_full_draws',
                   'B365H', 'B365D', 'B365A', 'league_id', 'country_id'
                  ]]

target = 'HTFTR'
IDcol = 'index'
predictors = [x for x in dtrain.columns if x not in [target, IDcol]]
dataset = dtrain.drop('HTFTR', axis=1).values

X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(dataset, classes, test_size=0.2)


def xgb_grid():
    rcParams['figure.figsize'] = 12, 4
    # Load Data


    cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
    ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                 'objective': 'binary:logistic'}
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params,
                                 scoring='precision', cv=5, n_jobs=7)

    optimized_GBM.fit(X_train, Y_train)
    print optimized_GBM.grid_scores_
    print optimized_GBM.cv_results_


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['HTFTR'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    print dtrain_predictions
    # Print model report:
    '''print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(Y_test, dtrain_predictions)
    print "Precision : %4g" % metrics.precision_score(Y_test, dtrain_predictions)
    print "Confusion Matrix: \n {0}".format(metrics.confusion_matrix(Y_test, dtrain_predictions))
    print "Classification report: \n {0}".format(metrics.classification_report(Y_test, dtrain_predictions))'''
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['HTFTR'], dtrain_predictions)
    print "Precision : %4g" % metrics.precision_score(dtrain['HTFTR'], dtrain_predictions)
    print "Confusion Matrix: \n {0}".format(metrics.confusion_matrix(dtrain['HTFTR'], dtrain_predictions))
    print "Classification report: \n {0}".format(metrics.classification_report(dtrain['HTFTR'], dtrain_predictions))
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['HTFTR'], dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #plt.show()

def xgb_train():
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=7,
        scale_pos_weight=1,
        silent=1,
        seed=27)
    modelfit(xgb1, dtrain, predictors)

def xgb_test_grid():
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=140, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=7, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1, scoring='precision', n_jobs=8, iid=False, cv=5)
    gsearch1.fit(dtrain[predictors], dtrain[target])
    print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

def test_xgb():
    #clf = XGBClassifier(max_depth=5, min_child_weight=3, nthread=7)
    kfold = model_selection.KFold(n_splits=2)
    cv_results = model_selection.cross_val_score(XGBClassifier(max_depth=7, min_child_weight=3, nthread=7, silent=0), X_train, Y_train, cv=kfold, scoring='precision')
    print cv_results
    msg = "%s: %f (%f)" % ("XGBoost: ", cv_results.mean(), cv_results.std())
    print(msg)


if __name__ == '__main__':
    #xgb_train()
    xgb_test_grid()
    #parameter_tune()


