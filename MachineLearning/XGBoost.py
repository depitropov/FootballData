import numpy as np
import pandas as pd
import sqlite3 as sq
from time import time

import xgboost as xgb
from sklearn import cross_validation, metrics
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams


rcParams['figure.figsize'] = 12, 4
# Load Data
conn = sq.connect('../database.db')
dataset = pd.read_sql_query('SELECT * FROM numpy_5', conn)
dataset['HTFTR'].replace(31, 1, inplace=True)
dataset.loc[dataset['HTFTR'] != 1, 'HTFTR'] = 0

# Half-time/Full-time combinations are 9

assert len(dataset['HTFTR'].unique()) == 2

# Fix Data
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
target = 'HTFTR'
IDcol = 'Index'


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=250):
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
    print dtrain_predictions
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    print dtrain_predprob

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['HTFTR'].values, dtrain_predictions)
    print "Precision : %.4g" % metrics.precision_score(dtrain['HTFTR'].values, dtrain_predictions)
    #print "Full result: %.4g" % metrics.classification_report(dtrain['HTFTR'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['HTFTR'], dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


def parameter_tune():

    # Find best HyperParameters
    predictors = [x for x in dataset.columns if x not in [target, IDcol]]
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=7,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb1, dataset, predictors)

if __name__ == '__main__':
    parameter_tune()


