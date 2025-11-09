from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


lr = LogisticRegression()

lr.fit(x_train_cv, y_train)
y_pred = lr.predict(x_test_cv)
print(lr)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, xticklabels=['predicted_anger', 'predicted_fear', 'predicted_joy', 'predicted_sad'],
            yticklabels=['actual_anger', 'actual_fear', 'actual_joy', 'actual_sad'],
            annot=True, fmt='d', annot_kws={'fontsize': 25}, cmap="YlGnBu");
plt.title(lr, fontsize=20)
plt.show()

# leaf_size_range = np.arange(1, 51, 5)
# n_neighbors_range = np.arange(1, 31, 5)
# p = [1, 2]
#
# hyperparameters = dict(leaf_size=leaf_size_range, n_neighbors=n_neighbors_range, p=p)
#
# lr2 = LogisticRegression()
#
# clf = GridSearchCV(lr2, hyperparameters, cv=5)
# best_model = clf.fit(x_train_cv, y_train)
# print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', best_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
# y_pred2 = clf.predict(x_test_cv)
# print(lr2)
# print(classification_report(y_test, y_pred2))