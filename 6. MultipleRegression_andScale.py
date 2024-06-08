# Multiple Regression is like Linear Regression
# - but with more than one independent values
# -- trying to predict using 2 or more variables
import numpy as np
# Problem: given a dataset with cars having
# - car brand name, model, engine-volume, car-weight and CO2 emissions
# -- it is required to predict the CO2 emissions
# --- based on the engine-volume and the car-weight

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# load the csv file into a pandas DataFrame object
cars_data_frame = pd.read_csv("./files/cars_data.csv", sep=";")
print(cars_data_frame)
print("axes =", cars_data_frame.axes)
print("ndim =", cars_data_frame.ndim)
print("size =", cars_data_frame.size)
print("shape =", cars_data_frame.shape)
print("dtypes =", cars_data_frame.dtypes)
print("at[7, Model] =", cars_data_frame.at[7, "Model"])
print("at[25, Volume] =", cars_data_frame.at[25, "Volume"])

# create DataFrame with the independent values
independent_v_df_X = cars_data_frame[["Weight", "Volume"]]
print(independent_v_df_X)
print(independent_v_df_X.axes)

# create DataFrame with the dependent values (y_true/target)
y_true = cars_data_frame["CO2"]
print(y_true)
print(y_true.axes)

# create our model for predicting using Linear Regression
linear_regression_model = linear_model.LinearRegression()
# train the model with the independent and dependent values (only values, no features names)
linear_regression_model.fit(independent_v_df_X.values, y_true.values)

# predict
X_test = np.array([[2300, 1300]])
print("X_test.shape :", X_test.shape)
print("X_test.ndim :", X_test.ndim)
pred_CO2_W2300 = linear_regression_model.predict(X_test)
print("predicted CO2:", pred_CO2_W2300)

# Coefficient
# - describes the relation of the dependent(target) value with the independent values used for the fitting
coefficients = linear_regression_model.coef_
print("coefficients =", coefficients)
print("coefficient[0] Weight =", coefficients[0])
print("coefficient[1] Volume =", coefficients[1])

# let's check the  predicted CO2 for a car of Weight 3300 and volume 1300
# - and verify the value in correspondence to the coefficient multiplication to the prev. value
X_test_W3300 = np.array([[3300, 1300]])
pred_CO2_W3300 = linear_regression_model.predict(X_test_W3300)
print("predicted CO2 Weight 3300:", pred_CO2_W3300)
print("predicted CO2 Weight 2300 + (1000 * coefficient[0]):", (pred_CO2_W2300 + (1000 * coefficients[0])))
# the printed values shows that it matches

# Scale Features
# - when the data has different measurement units: Kg, m, time
# -- the data can be scaled into new values in order to compare easier
# --- we will use the standardization method for scaling
# ---- formula: z = (x - u) / s
# ----- z, the new value
# ----- x, the original value
# ----- u, the mean
# ----- s, the standard deviation
# ------ the scikit-learn library has a StandardScaler object which transforms th data sets
scaler = StandardScaler()
scaled_independent_v_X = scaler.fit_transform(independent_v_df_X.values)
print("scaled_independent_v_X :", scaled_independent_v_X)

# create a new model for predicting with Linear Regression using scaled values
new_linear_regression_model = linear_model.LinearRegression()

# perform fitting(training) of the new Linear Regression Model using the scaled values
new_linear_regression_model.fit(scaled_independent_v_X, y_true.values)

# perform then prediction using the scaled values
scaled_X_test = scaler.transform(np.array([[2300, 1300]]))
print("scaled_X_test :", scaled_X_test)
pred_CO2_with_scaled = new_linear_regression_model.predict(scaled_X_test)
print("pred_CO2_with_scaled :", pred_CO2_with_scaled)
