from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

lr = LogisticRegression(solver ='lbfgs', multi_class='multinomial', max_iter=200)
nb = BernoulliNB()
dt = DecisionTreeClassifier()

estimators = []
estimators.append(('LR', lr))
estimators.append(('NB', nb))
estimators.append(('DT', dt))

def vot_hard_model():
    # Voting Classifier with hard voting
    vot_hard = VotingClassifier(estimators=estimators, voting='hard')
    vot_hard.fit(x_train_cv, y_train)

    return vot_hard

def vot_soft_model():
    # Voting Classifier with soft voting
    vot_soft = VotingClassifier(estimators=estimators, voting='soft')
    vot_soft.fit(x_train_cv, y_train)

    return vot_soft

# def accurcy():
#     lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)
#     nb = BernoulliNB()
#     dt = DecisionTreeClassifier()
#
#     print("Result of accuracy for each individual model")
#     print("==============================================================================")
#
#     lr.fit(x_train_cv, y_train)
#     y_pred = lr.predict(x_test_cv)
#     print("Multinomial Logistic Regression: " + str(accuracy_score(y_test, y_pred)))
#
#     nb.fit(x_train_cv, y_train)
#     y_pred = nb.predict(x_test_cv)
#     print("Naive Bayes: " + str(accuracy_score(y_test, y_pred)))
#
#     dt.fit(x_train_cv, y_train)
#     y_pred = dt.predict(x_test_cv)
#     print("Decision Tree: " + str(accuracy_score(y_test, y_pred)))


