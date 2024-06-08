import numpy as np
import matplotlib.pyplot as plt


# draw histogram of a uniform distribution with 20 bars
def plot_uniform_distribution():
    rand_array = np.random.uniform(0.0, 5.0, 250)
    plt.hist(rand_array, 20)
    plt.show()


# draw histogram of a normal distribution with 20 bars
def plot_normal_distribution():
    rand_array2 = np.random.normal(0.0, 5.0, 25000)
    plt.hist(rand_array2, 100)
    plt.show()


# Random data distribution
# - draw a scatter plot of points generated from 2 random arrays of 1000 numbers
# -- the x-axis array is a random normal distribution with mean 5.0, stdDev 1.0
# -- the y-axis array is a random normal distribution with mean 10.0, stdDev 2.0
def plot_scatter_with_normal_dist_points():
    x_axis = np.random.normal(5.0, 1.0, 100)
    y_axis = np.random.normal(10.0, 2.0, 100)
    plt.scatter(x_axis, y_axis)
    plt.show()


def execute(parameter):
    match parameter:
        case "uniform":
            plot_uniform_distribution()
        case "normal":
            plot_normal_distribution()
        case "scatter":
            plot_scatter_with_normal_dist_points()
        case _:
            print("\n\tNothing to execute!, parameter = ", parameter)


# execute("uniform")
# execute("normal")
execute("scatter")
