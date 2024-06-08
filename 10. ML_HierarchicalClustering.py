# Hierarchical Clustering
# - Unsupervised Learning method for clustering data points
# -- Model doesn't need training, Target variable is not needed

# --- Aglomerative Hierarchical Clustering
# ---- bottom-up
# ----- first each data-point has its own cluster
# ------ then clusters are joined based on shortest distance
# ------- repeated until all clusters are inside one large cluster

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (rand_score, mutual_info_score, adjusted_rand_score,
                             adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)

zoo_data_frame = pd.read_csv("./files/zoo_data.csv", sep=";")
print(zoo_data_frame)
# [101 rows x 18 columns]
print(zoo_data_frame.axes)

print(zoo_data_frame.isnull().sum().sum())
# If 0 then our datasets does not have null values

# check how many samples are present per labeled cluster
print(zoo_data_frame.groupby(["type"]).count().sort_values(by="type", ascending=True))

# separate labels from the whole data
ground_truth_labels = zoo_data_frame["type"]
print("ground_truth_labels :", ground_truth_labels)

# separate data features from the labels
zoo_df_features = zoo_data_frame.drop(["animal", "type"], axis=1)
print("shape: {}".format(zoo_df_features.shape))
print("columns: {}".format(zoo_df_features.columns))

# lists with the Clustering parameters: linkages and metrics
linkages = ["ward", "complete", "average", "single"]
metrics = ["euclidean", "l1", "l2", "manhattan", "cosine"]

# open a *.csv file to write the clustering evaluation results
evaluation_file = open("./files/clustering_evaluation.csv", "w")
evaluation_file.write("Clustering, RandIndex, AdjRandIndex, MutualInfo, AdjMutualInfo, V-Measure, "
                      "Fowlkes-Mallows, Silhouette, Calinski-Harabasz, Davies-Bouldin\n")


# Metrics for Clustering:
# - extrinsic measures, using ground truth labels
# ---- Rand Index (rand_score, adjusted_rand_score)
# ---- Mutual Information (mutual_info_score, adjusted_mutual_info_score)
# ---- V-measure (v_measure_score)
# ---- Fowlkes-Mallows Score
# - intrinsic measures, without ground truth labels
# ---- Silhouette coefficient (silhouette_score)
# ---- Calinski-Harabasz Index (calinski_harabasz_score)
# ---- Davies-Bouldin Index (davies_bouldin_score)
def evaluate_model(true_labels, pred_labels, data, row_header="_"):
    rand_index = rand_score(true_labels, pred_labels)
    rand_index = "{:2.3%}".format(rand_index)
    print(rand_index)

    adjust_rand_index = adjusted_rand_score(true_labels, pred_labels)
    adjust_rand_index = "{:2.3%}".format(adjust_rand_index)
    print(adjust_rand_index)

    mutual_info = mutual_info_score(true_labels, pred_labels)
    mutual_info = "{:2.3%}".format(mutual_info)
    print(mutual_info)

    adjust_mutual_info = adjusted_mutual_info_score(true_labels, pred_labels)
    adjust_mutual_info = "{:2.3%}".format(adjust_mutual_info)
    print(adjust_mutual_info)

    v_measure = v_measure_score(true_labels, pred_labels)
    v_measure = "{:2.3%}".format(v_measure)
    print(v_measure)

    fowlkes_mallows = fowlkes_mallows_score(true_labels, pred_labels)
    fowlkes_mallows = "{:2.3%}".format(fowlkes_mallows)
    print(fowlkes_mallows)

    silhouette = silhouette_score(data, pred_labels)
    silhouette = "{:.5f}".format(silhouette)
    print(silhouette)

    calinski_harabasz = calinski_harabasz_score(data, pred_labels)
    calinski_harabasz = "{:.5f}".format(calinski_harabasz)
    print(calinski_harabasz)

    davies_bouldin = davies_bouldin_score(data, pred_labels)
    davies_bouldin = "{:.5f}".format(davies_bouldin)
    print(davies_bouldin)
    row_string = ("{}, {}, {}, {}, {}, "
                  "{}, {}, {}, {}, {}\n").format(row_header, rand_index, adjust_rand_index,
                                                 mutual_info, adjust_mutual_info, v_measure,
                                                 fowlkes_mallows, silhouette, calinski_harabasz,
                                                 davies_bouldin)
    return row_string


for linkage in linkages:
    for metric in metrics:
        if linkage == "ward" and metric != "euclidean":
            continue
        row_heading = "Linkage {} - Metric {}".format(linkage, metric)
        print(row_heading)

        # Create the Hierarchical Clustering Model
        # - with 7 clusters as the animal types available
        model = AgglomerativeClustering(n_clusters=7, linkage=linkage, metric=metric)
        # -- Fit and Predict the Unsupervised Learning Model, obtain the predicted labels
        predicted_labels = model.fit_predict(zoo_df_features)
        # predicted_labels = model.labels_
        # --- Evaluate the Model
        evaluation_row = evaluate_model(ground_truth_labels, predicted_labels,
                                        zoo_df_features, row_header=row_heading)
        print(evaluation_row)
        evaluation_file.write(evaluation_row)


evaluation_file.close()
# after checking the csv file with all the evaluation results
# - I see that the best scores are obtained using
# -- Linkage "complete" and Metric "manhattan"
final_model = AgglomerativeClustering(n_clusters=7, linkage="complete", metric="manhattan")
final_model.fit(zoo_df_features)
final_predicted_labels = final_model.labels_

print(np.array(ground_truth_labels))
print(final_predicted_labels)
