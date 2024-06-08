# Imbalanced data classification
# -- has the distribution of class labels skewed in the training dataset

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             roc_auc_score, roc_curve)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier


def evaluate_classifier(target_test, target_pred, target_names):

    print("accuracy: {:2.3%}".format(accuracy_score(target_test,
                                                    target_pred)))
    print("precision: {:2.3%}".format(precision_score(target_test,
                                                      target_pred,
                                                      average="weighted")))
    print("sensitivity_recall: {:2.3%}".format(recall_score(target_test,
                                                            target_pred,
                                                            average="weighted")))
    print("f1_score: {:2.3%}".format(f1_score(target_test, target_pred,
                                              average="weighted")))
    print(classification_report(target_test, target_pred,
                                target_names=target_names))


# Set a random seed
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# load the Haberman Breast Cancer dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
dataframe = pd.read_csv(url, header=None)
# get the values
values = dataframe.values
X, y = values[:, :-1], values[:, -1]

# gather details
n_rows = X.shape[0]
n_cols = X.shape[1]
classes = np.unique(y)
n_classes = len(classes)
# summarize
print('N Examples: %d' % n_rows)
print('N Inputs: %d' % n_cols)
print('N Classes: %d' % n_classes)
print('Classes: %s' % classes)
print('Class Breakdown:')

# class breakdown
breakdown = ''
for c in classes:
    total = len(y[y == c])
    ratio = (total / float(len(y))) * 100
    print(' - Class %s: %d (%.5f%%)' % (str(c), total, ratio))


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=random_seed)

# Create a base classifier (DecisionTreeClassifier)
base_classifier = DecisionTreeClassifier(random_state=random_seed)

# Create a BaggingClassifier
bagging_classifier = BalancedBaggingClassifier(base_classifier, random_state=random_seed)

# define the parameter grid dictionary to search over
param_grid = [{"bag__n_estimators": [10, 20, 40, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 2000, 3000]}]
# {"bag__n_estimators": [50, 100, 200, 300, 400, 500],
#  "bag__estimator__class_weight": [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5},
#   {0: 1, 1: 7}, {0: 1, 1: 9}]},

# cross-validation method can be beneficial
# - in scenarios where data has class imbalance
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

# create under-sampler, removes samples from the majority class
under_sampler = RandomUnderSampler(sampling_strategy="auto", random_state=random_seed)
# create over-sampler, adds samples auto generated
over_sampler = RandomOverSampler(sampling_strategy="auto", random_state=random_seed)
smote = SMOTE(sampling_strategy="auto", random_state=random_seed)
svm_smote = SVMSMOTE(sampling_strategy="auto", random_state=random_seed)

# Create a pipeline with the under-sampler
pipeline = Pipeline([
    ('svm_smote', svm_smote),
    ('bag', bagging_classifier)
])

grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_cv, verbose=3,
                           scoring="roc_auc", n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# print the best parameters identified by GridSearchCV
print("Best Parameters: ", grid_search.best_params_, grid_search.scoring, grid_search.best_score_)

# get the best model indicated by GridSearchCV
best_model = grid_search.best_estimator_

# get class probabilities for each sample on the test set
y_proba = best_model.predict_proba(X_test)
# get the probabilities for the positive class (2nd column)
y_positive_proba = y_proba[:, 1]

# predict using the test set
y_pred = best_model.predict(X_test)
print("y_pred", y_pred)

evaluate_classifier(y_test, y_pred, ["Survived 5", "Died within 5"])

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_positive_proba)
print("ROC AUC Score: {:2.3%}".format(roc_auc))
# RESULTS:
# 1.) n_estimators 500, class_weight None
# --- ROC_AUC 69%, accuracy 68,4%, precision 65,2%
# 2.) other result
# --- Train {estimator__class_weight: None, n_estimators: 300} roc_auc 0.7057186994686996
# ---- Test ROC AUC Score: 61.364%
# 3.) Best until now
# --- Train {n_estimators: 500} roc_auc 0.6629123900293254
# ---- Test(30%Data) ROC AUC Score: 69.172%
# ----- changing stratifiedKFold from 5 to 10 does not change much
# 4.) Under-sampling technique (RandomUnderSampler, BalancedBaggingClassifier, imblearn.pipeline)
# --- Train bag__n_estimators: 500 roc_auc 0.7124652777777778
# ---- Test(30%Data) ROC AUC Score: 64.423%
# 5.) Over-sampling technique (RandomOverSampler, BalancedBaggingClassifier, imblearn.pipeline)
# --- Train bag__n_estimators:50 roc_auc 0.6636458333333334
# ---- Test(30%Data) ROC AUC Score: 70.309%
# 6.) SMOTE Over-sampling technique (SMOTE, BalancedBaggingClassifier, imblearn.pipeline)
# --- Train bag__n_estimators:20 roc_auc 0.6879861111111112
# ---- Test(30%Data) ROC AUC Score: 65.559%
# 7.) SVMSMOTE Over-sampling technique (SVMSMOTE, BalancedBaggingClassifier, imblearn.pipeline)
# --- Train bag__n_estimators:100 roc_auc 0.6786458333333332
# ---- Test(30%Data) ROC AUC Score:  65.239%

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_positive_proba, pos_label=2)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.3f})")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
