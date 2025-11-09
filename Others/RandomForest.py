from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, precision_score, roc_auc_score

from readDataset import x_test, x_train, y_train, y_test, x_train_cv, x_test_cv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

rf = RandomForestClassifier()

rf.fit(x_train_cv, y_train)
y_pred = rf.predict(x_test_cv)
print(rf)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, xticklabels=['predicted_anger', 'predicted_fear', 'predicted_joy', 'predicted_sad'],
            yticklabels=['actual_anger', 'actual_fear', 'actual_joy', 'actual_sad'],
            annot=True, fmt='d', annot_kws={'fontsize': 25}, cmap="YlGnBu");
plt.title(rf, fontsize=20)
plt.show()