# Bootstrap Aggregation
# --- Bagging Classifier

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (accuracy_score, precision_score, make_scorer,
                             recall_score, f1_score, classification_report)


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

# load the iris dataset
iris = load_iris()
print("target names:", iris["target_names"])
# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"],
                                                    test_size=0.4, random_state=random_seed)

# Now let's create a Pipeline with tuples
# - 1 transformer: StandardScaler
# -- 1 final estimator: BaggingClassifier
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("bag", BaggingClassifier(random_state=random_seed,
                                               oob_score=False))])
# define the parameter grid dictionary to search over
param_grid = [{"bag__n_estimators": [2, 3, 4, 5, 6, 10, 11, 12, 14, 16, 20, 21, 25, 30, 36, 40, 45, 50, 100, 200, 500]}]

# create a StratifiedKFold object for cross-validation
cross_validation = StratifiedKFold(n_splits=5, shuffle=True,
                                   random_state=random_seed)

# create a GridSearchCV for searching the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=cross_validation,
                           n_jobs=-1, verbose=3,
                           scoring=make_scorer(precision_score, average="weighted",
                                               zero_division=0))

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# print the best parameters found
print("Best Parameters :", grid_search.best_params_)

# Get the Best Model
best_model = grid_search.best_estimator_

# perform predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the precision of the model
evaluate_classifier(y_test, y_pred, iris["target_names"])

# Extract the results from the grid search
results = pd.DataFrame(grid_search.cv_results_)
# Plot the results
print(grid_search.cv_results_)
print(results.columns)

# Plot the results
plt.plot(results['param_bag__n_estimators'], results['mean_test_score'], marker='o')
plt.title('GridSearchCV Results for BaggingClassifier')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Mean Test Score (Precision)')
plt.show()

# plot the first Decision Tree from the final voting
plot_tree(best_model.named_steps["bag"].estimators_[0], feature_names=iris["feature_names"])
plt.show()
