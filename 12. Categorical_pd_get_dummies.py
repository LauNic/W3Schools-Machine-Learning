# Categorical data represented by strings
# - can be transformed into numerical, 2 ways:
# -- 1.) pandas get_dummies transforms category into columns with value 0, 1
# -- 2.) category-label encoding with numerical value using a dictionary

# in this program I will use the first way
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Set a random seed
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# get the data from the csv file
cars_df = pd.read_csv("./files/cars_data.csv", sep=";")
print(cars_df)

# get the independent numerical features separately
cars_volume_weight = cars_df[["Volume", "Weight"]]
print(cars_volume_weight)

# transform categorical values into columns with value 0 and 1
car_and_model = pd.get_dummies(cars_df[["Car", "Model"]], dtype=float)
print(car_and_model)

# put together the numerical with the transformed dummies
X = pd.concat([cars_volume_weight, car_and_model], axis=1)
print(X)
y = cars_df["CO2"]
print(y)

# Split the data into training and testing (only 10% because the dataset is very small)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=random_seed)

# Now let's create a Pipeline with tuples
# - 1 transformer: StandardScaler
# -- 1 final estimator: LinearRegression
# --- the Pipeline will be used by a GridSearchCV
# ---- along with a parameter grid dictionary
# ----- and a StratifiedKFold with 5 splits, shuffling
pipeline = Pipeline([("scaler", StandardScaler()),
                     ("ridge", Ridge(random_state=random_seed))])

# define the parameter grid dictionary to search over
param_grid = [{"ridge__alpha": np.logspace(-3, 3, 7)}]

# Data is too small => Not necessary to  create a StratifiedKFold object for cross-validation
# cross_valid = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_seed)

# create a GridSearchCV object for searching the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1,
                           scoring="neg_mean_squared_error", verbose=1)

# fit the GridSearchCV using the training data
grid_search.fit(X_train, y_train)

# the best parameters found
print("Best Params :", grid_search.best_params_)

# get the Best Model
best_model = grid_search.best_estimator_

# predict with the Best Model using the test data
y_pred = best_model.predict(X_test)

# evaluate the model with metric mean squared error on the test set predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error : ", mse)
print("R2 score: ", r2_score(y_test, y_pred))
