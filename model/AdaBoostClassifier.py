import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def adaBoost_model():
    adaBoost = AdaBoostClassifier()

    adaBoost.fit(x_train_cv, y_train)

    return adaBoost

def adaBoost_tuning_model():
    # Best n_estimators: 161
    # Best learning_rate: 0.7000000000000001
    # Best algorithm: SAMME.R
    n_estimators = np.arange(1, 201, 20) #1, 21, 41, 61 ... 181
    learning_rate = np.arange(0.1, 1, 0.1) #0.1, 0.2, 0.3, 0.4, 0.5 ..... 0.9
    algorithm = ['SAMME', 'SAMME.R']

    hyperparameters = dict(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)

    adaBoost = AdaBoostClassifier()

    clf = GridSearchCV(adaBoost, hyperparameters, cv=5)
    best_model = clf.fit(x_train_cv, y_train)

    print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
    print('Best learning_rate:', best_model.best_estimator_.get_params()['learning_rate'])
    print('Best algorithm:', best_model.best_estimator_.get_params()['algorithm'])

    return clf