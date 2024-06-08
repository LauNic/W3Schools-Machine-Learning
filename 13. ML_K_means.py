# K-means - Unsupervised learning method
# -- used for clustering data points into K clusters
# ---- minimizing the variance in each cluster

# Find best value for K using the elbow method:
# - each data point is randomly assigned to one of the K clusters
# -- compute the centroid/center of each cluster
# --- reassign each data point to the cluster with nearest centroid
# ---- repeat process until cluster assignment does not change anymore

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# load the iris dataset
iris = datasets.load_iris()
# put the data in a pandas dataframe
data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
# select only 2 features for simplicity and visualization
X = data[["sepal length (cm)", "sepal width (cm)"]]
print(X)

# utilize the elbow method to visualize the inertias for K values
inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init="auto", random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertias, marker="o")
plt.title("Elbow method")
plt.xlabel("N. Clusters")
plt.ylabel("Inertia")

# for the iris dataset we know there are 3 clusters
k_means = KMeans(n_clusters=3, n_init="auto", random_state=42)
k_means.fit(X)

# add a cluster labels to the DataFrame
data["cluster"] = k_means.labels_
print(data["cluster"])

print(k_means.cluster_centers_)
print(k_means.cluster_centers_[:, 0])

# visualize the clusters
plt.figure()
x_points = data["sepal length (cm)"]
y_points = data["sepal width (cm)"]
colors = data["cluster"]
plt.scatter(x_points, y_points, c=colors)

x_centroids = k_means.cluster_centers_[:, 0]
y_centroids = k_means.cluster_centers_[:, 1]
plt.scatter(x_centroids, y_centroids, marker="X", s=100, c="red", label="Centroids")
plt.legend()
plt.show()
