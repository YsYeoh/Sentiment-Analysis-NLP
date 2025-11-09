import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Best leaf_size: 1, Best p: 2, Best n_neighbors: 29
# knn = KNeighborsClassifier(leaf_size=1, p=2, n_neighbors=29)
def knn_model():
    knn = KNeighborsClassifier()

    knn.fit(x_train_cv, y_train)

    return knn

def knn_tuning_model():
    leaf_size_range = np.arange(1, 51, 1)
    n_neighbors_range = np.arange(1, 31, 1)
    p = [1, 2]

    hyperparameters = dict(leaf_size=leaf_size_range, n_neighbors=n_neighbors_range, p=p)

    knn2 = KNeighborsClassifier()

    clf = GridSearchCV(knn2, hyperparameters, cv=5)
    best_model = clf.fit(x_train_cv, y_train)

    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

    return clf


