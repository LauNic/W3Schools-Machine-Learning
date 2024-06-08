# Polynomial Regression can be used if the data does not fit linear regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 2 arrays of car hour-time of passing and of car speed
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

plt.scatter(x, y)
# plt.show()
# seeing the points on the graph, I understand that there is a polynomial function/regression pattern

# generate array of polynomial coefficients
coefficients = np.polyfit(x, y, 3)

# construct the predicting polynomial function using the array of polynomial coefficients
polynomial_function = np.poly1d(coefficients)

# array with the points that will be displayed on the x-axis
x_axis_polynomial = np.linspace(1, 22, 10000)

plt.plot(x_axis_polynomial, polynomial_function(x_axis_polynomial))
plt.show()

# calculate the R Squared (regression score/ coefficient of determination)
# - checking y_true(the y variable in the program) versus y_pred(executing polynomial function on the x)
# -- returns the relationship score, 0 means no relationship and 1 means 100% related
regression_score = r2_score(y, polynomial_function(x))
print("Regression score = ", regression_score)

# calculate the mean squared error
regression_loss = mean_squared_error(y, polynomial_function(x))
print("Regression loss = ", regression_loss)

# predict the speed of a car at hour-time 17
pred_speed = polynomial_function(17)
print("Predicted speed = ", pred_speed)
