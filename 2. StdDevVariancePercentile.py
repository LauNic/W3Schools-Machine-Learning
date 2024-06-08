import numpy as np

# Python List
speed = [86, 87, 88, 86, 87, 85, 86]

# Standard Deviation is a number describing how spread the value are
std_dev = np.std(speed)
print("std_dev = {:.5f}".format(std_dev))

# Standard Deviation is the square root of the Variance
# which is another value indicating how spread the values are
variance = np.var(speed)
print("variance = {:.5f}".format(variance))
print("std_dev^2 = variance = {:.5f}".format(std_dev*std_dev))

# Percentile
# Given an array(python list) of numbers representing peoples numbers
# we can obtain the maximum number of that percent
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]

percentile = np.percentile(ages, 75)
print("percentile = ", percentile)
