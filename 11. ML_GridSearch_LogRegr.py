# Run Logistic Regression fit and predict in multiple rounds
# -- with GridSearchCV trying different parameters for the estimator

import numpy as np
import random
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import make_scorer, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Set a random seed
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


# Model Evaluation through sklearn.metrics
def evaluate_logreg_model(labels_test, labels_pred, target_names):
    # - how often the model is correct
    accuracy = metrics.accuracy_score(labels_test, labels_pred)
    print("accuracy : {:2.3%}".format(accuracy))

    # - of the predicted positives what percentage is truly positive
    precision = metrics.precision_score(labels_test, labels_pred, average="weighted")
    print("precision : {:2.3%}".format(precision))

    # - how good is the model at predicting positives
    sensitivity_recall = metrics.recall_score(labels_test, labels_pred, average="weighted")
    print("sensitivity_recall : {:2.3%}".format(sensitivity_recall))

    # - harmonic mean of precision and sensitivity_recall
    f1_score = metrics.f1_score(labels_test, labels_pred, average="weighted")
    print("f1_score : {:2.3%}".format(f1_score))

    print(metrics.classification_report(labels_test, labels_pred, target_names=target_names))


iris = datasets.load_iris()
print("iris[data] =", iris["data"])
print("iris[target] =", iris["target"])
print("iris[target_names] =", iris["target_names"])

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"],
                                                    test_size=0.4, random_state=random_seed)

# Feature pre-processing, Scaling
scaler = StandardScaler()
# Fit the scaler on the training data
scaler.fit(X_train)
# Transform the training and testing data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the Logistic Regression Model
logistic_regression_model = LogisticRegression(random_state=random_seed, max_iter=10000, C=10,
                                               class_weight=None, fit_intercept=True,
                                               penalty="l1", solver="saga")

# Perform the training using the model
logistic_regression_model.fit(X_train_scaled, y_train)

# predict using the trained model and the testing dataset
y_pred = logistic_regression_model.predict(X_test_scaled)
evaluate_logreg_model(y_test, y_pred, iris["target_names"])
print(" END of first RUN")
# -- 60% data for training 40% for testing

# Now let's create a Pipeline with tuples
# - 1 transformer: StandardScaler
# -- 1 final estimator: LogisticRegression
# --- the Pipeline will be used by a GridSearchCV
# ---- along with a parameter grid dictionary
# ----- and a StratifiedKFold with 5 splits, shuffling
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(random_state=random_seed))])

# define the parameter grid dictionary to search over
param_grid = [{"logreg__C": [0.01, 0.05, 0.1, 0.5, 1, 10, 20],
               "logreg__penalty": ["l1", "l2"],
               "logreg__solver": ["saga", "liblinear"],
               "logreg__max_iter": [4000, 5000, 6000],
               "logreg__fit_intercept": [True, False],
               "logreg__class_weight": [None, "balanced"]},
              {"logreg__C": [0.01, 0.05, 0.1, 0.5, 1, 10, 20],
               "logreg__penalty": ["l2"],
               "logreg__solver": ["newton-cg", "sag", "lbfgs", "newton-cholesky"],
               "logreg__max_iter": [4000, 5000, 6000],
               "logreg__fit_intercept": [True, False],
               "logreg__class_weight": [None, "balanced"]}]

# create a StratifiedKFold object for cross-validation
cross_valid = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

# create a GridSearchCV object for searching the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=cross_valid, n_jobs=-1, verbose=1,
                           scoring=make_scorer(precision_score, average="weighted", zero_division=0))

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# print the best parameters found
print("Best Parameters :", grid_search.best_params_)

# Get the Best Model
best_model = grid_search.best_estimator_

# perform predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the precision of the model
# -- using cross-validated predictions on the test set
evaluate_logreg_model(y_test, y_pred, iris["target_names"])
# Best Parameters :  {'logreg__C': 10, 'logreg__class_weight': None, 'logreg__fit_intercept': True,
# (with no warnings)  'logreg__max_iter': 4000, 'logreg__penalty': 'l2', 'logreg__solver': 'saga'}
# accuracy : 98.333%
# precision : 98.417%
