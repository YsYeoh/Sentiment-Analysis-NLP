from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

svm = SVC()
svm = SVC(C=100, gamma=0.01, kernel='sigmoid')

svm.fit(x_train_cv, y_train)
y_pred = svm.predict(x_test_cv)
print(svm)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, xticklabels=['predicted_anger', 'predicted_fear', 'predicted_joy', 'predicted_sad'],
            yticklabels=['actual_anger', 'actual_fear', 'actual_joy', 'actual_sad'],
            annot=True, fmt='d', annot_kws={'fontsize': 25}, cmap="YlGnBu");
plt.title(svm, fontsize=20)
plt.show()

def tuning():
    # Result
    # Best C: 100
    # Best gamma: 0.01
    # Best kernel: sigmoid

    C = [0.1, 1, 10, 100]
    gamma = [1, 0.1, 0.01, 0.001]
    kernel = ['rbf', 'poly', 'sigmoid']

    hyperparameters = dict(C=C, gamma=gamma, kernel=kernel)

    svm2 = SVC()

    clf = GridSearchCV(svm2, hyperparameters, cv=5)
    best_model = clf.fit(x_train_cv, y_train)
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    print('Best gamma:', best_model.best_estimator_.get_params()['gamma'])
    print('Best kernel:', best_model.best_estimator_.get_params()['kernel'])
    y_pred2 = clf.predict(x_test_cv)
    print(svm2)
    print(classification_report(y_test, y_pred2))

    cm1 = confusion_matrix(y_test, y_pred2)
    sns.heatmap(cm1, xticklabels=['predicted_anger', 'predicted_fear', 'predicted_joy', 'predicted_sad'],
                yticklabels=['actual_anger', 'actual_fear', 'actual_joy', 'actual_sad'],
                annot=True, fmt='d', annot_kws={'fontsize': 25}, cmap="YlGnBu");
    plt.title(svm, fontsize=20)
    plt.show()