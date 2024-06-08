# (KNN) K-nearest-neighbors
# - supervised ML algorithm
# -- can be used for both classification or regression
# --- often used in:
# ---- missing value imputations, intrusion detection, pattern recognition
# ----- K is the number of nearest neighbors
# ------ larger values of K are more robust to outliers
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Set a random seed
random_seed = 42

# load the iris dataset and print data
iris = datasets.load_iris()
print(iris["target"])
print(iris["data"])
print(iris["data"][:, -2:])
# select only 2 features for simplicity and visualization
iris_petal_features = iris["data"][:, -2:]

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(iris_petal_features, iris["target"],
                                                    test_size=0.3, random_state=random_seed)

print("X_train")
print(X_train)
print("X_train[:, 0]")
print(X_train[:, 0])
print("X_train[:, 1]")
print(X_train[:, 1])

# define new test points for the plot
new_points = np.array([[4.0, 1.4], [1.5, 0.5], [5.5, 2.4]], float)
print("type(new_point): ", type(new_points))
print("shape(new_point): ", new_points.shape)

# visualize the clusters of points for iris petal dimensions
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
x_points = X_train[:, 0]
y_points = X_train[:, 1]
train_colors = y_train
plt.subplot(2, 2, 1)
plt.scatter(x_points, y_points, c=train_colors)
plt.xlabel("petal length cm")
plt.ylabel("petal width cm")
plt.title("Before Training")
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", s=12, c="red", label="Test data")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x_points, y_points, c=train_colors)
plt.xlabel("petal length cm")
plt.ylabel("petal width cm")
plt.title("Before Training - 1 point prediction")
plt.scatter(new_points[:, 0], new_points[:, 1], marker="o", s=12, c="blue", label="Test points")
plt.legend()

#  Define  the model
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
# Train the model
knn.fit(X_train, y_train)

# Predict the class of the new point
new_point_pred = knn.predict(new_points)
print("new_point_pred:", new_point_pred)

# Predict the classes for the test data
y_pred = knn.predict(X_test)
print("y_pred:", y_pred)

# 3.) draw the scatter plot with predicted classes
plt.subplot(2, 2, 3)
plt.scatter(x_points, y_points, c=train_colors)
plt.xlabel("petal length cm")
plt.ylabel("petal width cm")
plt.title("After Training")
plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", s=25, c=y_pred)

# 4.) draw the scatter plot with predicted classes
plt.subplot(2, 2, 4)
plt.scatter(x_points, y_points, c=train_colors)
plt.xlabel("petal length cm")
plt.ylabel("petal width cm")
plt.title("After Training - 1 point prediction")
plt.scatter(new_points[:, 0], new_points[:, 1], marker="x", s=25, c=new_point_pred)

plt.show()
