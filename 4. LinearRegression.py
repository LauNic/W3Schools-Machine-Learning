# Regression is used to find relationship between variables
# - In Machine Learning and in statistical modelling
# -- this relationship is used to predict outcome of future events

# Linear Regression uses the relationship between data-points
# - to draw a straight line through them,
# -- this line is used to predict future values

import matplotlib.pyplot as plt
from scipy import stats

# 2 arrays of car years and speeds
x_axis = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y_axis = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# obtain method for key values of linear regression
slope, intercept, r, p, std_err = stats.linregress(x_axis, y_axis)


# new function using slope and intercept, function used to predict
# to return the y_axis values for the linear regression line
def linear_regression_function(x):
    return slope * x + intercept


# map runs the function for each value of the array and the results are saved in the list
linear_regression_y_axis = list(map(linear_regression_function, x_axis))

plt.scatter(x_axis, y_axis)
plt.plot(x_axis, linear_regression_y_axis)
plt.show()

# Relationship r ranges between -1.0 and 1.0, 0 means no relation 1,-1 means 100% relation
if abs(r) > 0.7:
    print("There is a relationship between points, r = ", r)
    print("We can predict future values")

# predict the speed for a car of 10 years
pred_speed = linear_regression_function(10)
print("Predicted speed = ", pred_speed)
