from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def nb_model():
    nb = BernoulliNB()

    nb.fit(x_train_cv, y_train)

    return nb

def nb_tuning_model():
    # Best fit_prior: False
    # Best alpha: 0.5299999999999996
    # Best binarize: 0.0
    fit_prior = [True, False]
    alpha = np.arange(1.0, 0, -0.01)
    binarize = np.arange(0.0, 1.0, 0.01)

    hyperparameters = dict(fit_prior=fit_prior, alpha=alpha, binarize=binarize)

    nb2 = BernoulliNB()
    clf = GridSearchCV(estimator=nb2, param_grid=hyperparameters, cv=10)
    best_model = clf.fit(x_train_cv, y_train)

    print('Best fit_prior:', best_model.best_estimator_.get_params()['fit_prior'])
    print('Best alpha:', best_model.best_estimator_.get_params()['alpha'])
    print('Best binarize:', best_model.best_estimator_.get_params()['binarize'])

    return clf

