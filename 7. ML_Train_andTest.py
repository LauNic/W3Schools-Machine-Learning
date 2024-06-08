# Train/Test
# - a method to measure the accuracy of the model
# --  the data set is split into 80% train data and 20% test data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
# create a normal distribution array with mean 3, std-dev 1 and 100 samples
x = np.random.normal(3, 1, 100)
# create a normal distribution array with mean 150, std-dev 40 and 100 samples
y = np.random.normal(150, 40, 100) / x
# x-axis minutes before purchase, y-axis spent money amount

# keep 80% of the original data for training
train_x = x[:int(x.size*0.8)]
train_y = y[:int(y.size*0.8)]
# keep 20% of the original data for testing
test_x = x[int(x.size*0.8):]
test_y = y[int(y.size*0.8):]

# generate array of polynomial coefficients
coefficients = np.polyfit(train_x, train_y, 4)
# construct the predicting polynomial function using the array of polynomial coefficients
polynomial_function = np.poly1d(coefficients)
# array with the points that will be displayed on the x-axis
x_axis_polynomial = np.linspace(0, 6, 100)
# draw polynomial regression through the data points
plt.plot(x_axis_polynomial, polynomial_function(x_axis_polynomial))

# calculate the R Squared (regression score/ coefficient of determination)
# - checking y_true(the y variable in the program) versus y_pred(executing polynomial function on the x)
# -- returns the relationship score, 0 means no relationship and 1 means 100% related
y_true = train_y
y_pred = polynomial_function(train_x)
regression_score_train = r2_score(y_true, y_pred)
print("Training Regression score = ", regression_score_train)
# Training Regression score =  0.79886455446298
regression_score_test = r2_score(test_y, polynomial_function(test_x))
print("Testing Regression score = ", regression_score_test)
# Testing Regression score =  0.8086921460343566
# the polynomial function model has a good r2 score for both train and test data

# the testing set looks similar to the training set
plt.scatter(train_x, train_y, c="blue")
plt.scatter(test_x, test_y, c="red")

# predict money spent for 5.5 minutes in store
print("predict money spent for 5.5 minutes in store :", polynomial_function(5.5))
plt.plot(5.5, polynomial_function(5.5), 'go', markersize=9)
plt.plot([5.5, 5.5], [0, polynomial_function(5.5)], 'g:')
plt.plot([0, 5.5], [polynomial_function(5.5), polynomial_function(5.5)], 'g:')

plt.show()
