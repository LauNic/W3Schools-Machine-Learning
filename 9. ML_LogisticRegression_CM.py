# Logistic Regression
# - solves classification problems
# -- by predicting categorical outcomes
# --- unlike Linear Regression predictions for continuous outcomes
# ---- e.g. predicting if a tumor is malign or benign

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# initialize some package settings
sns.set(style="whitegrid", color_codes=True, font_scale=1.0)

# load the csv file into a pandas DataFrame object
breast_cancer_df = pd.read_csv("./files/Breast_Cancer_Wisconsin_data.csv", sep=",")
print(breast_cancer_df)

# Decision trees needs that all data is numerical
diagnosis_dictionary = {"M": 1, "B": 0}
# use the pandas map() method to transform value into numerical
breast_cancer_df["Diagnosis"] = breast_cancer_df["Diagnosis"].map(diagnosis_dictionary)
breast_cancer_df = breast_cancer_df.drop(["ID"], axis=1)
print(breast_cancer_df)
print(breast_cancer_df.groupby(["Diagnosis"]).count().sort_values(by="Diagnosis", ascending=True))

# separate  feature columns from the target column
feature_columns = breast_cancer_df.keys().drop(["Diagnosis"]).values
breast_cancer_df_features = breast_cancer_df[feature_columns]
print(breast_cancer_df_features)
print(breast_cancer_df_features.axes)
# dataframe with only the target values
breast_cancer_df_target = breast_cancer_df["Diagnosis"]
print(breast_cancer_df_target)
print(breast_cancer_df_target.axes)

# generate correlation matrix and check for multi-collinearity
# - after first check I see multi-collinearity for radius, perimeter and area
# -- remove area, perimeter, all *3 features as correlated with the *1
breast_cancer_df_features = breast_cancer_df_features.drop(["area1", "area2", "area3", "perimeter1",
                                                            "perimeter2", "perimeter3", "radius3",
                                                            "texture3", "concave_points3", "concavity3",
                                                            "compactness3", "smoothness3", "symmetry3",
                                                            "fractal_dimension3", "concavity1", "concave_points1",
                                                            "concavity2", "concave_points2"], axis=1)
correlation_matrix = breast_cancer_df_features.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(correlation_matrix, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
# Set figure size
f, ax = plt.subplots(figsize=(10, 10))
# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()
# plt.show()
# after removing the highly correlated features the model looks ok

# Normalize the data
scaler = StandardScaler()
breast_cancer_df_features = scaler.fit_transform(breast_cancer_df_features.values)

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_df_features, breast_cancer_df_target,
                                                    test_size=0.3, random_state=40)

# create the Logistic Regression Model
logistic_regression_model = linear_model.LogisticRegression(random_state=40)

# perform the training using the model
logistic_regression_model.fit(X_train, y_train)

# predict using the trained model and the testing dataset
y_pred = logistic_regression_model.predict(X_test)
print(y_pred)

# Confusion Matrix
# - in classification problems
# -- it is a table for showing errors in the model
# --- rows represent actual classes, the true outcomes
# ---- columns represent the predictions
# create a confusion matrix from the prediction made by the Logistic Regression Model
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                          display_labels=[False, True])
confusion_matrix_display.plot()

# Model Evaluation through sklearn.metrics:
# - how often the model is correct
accuracy = metrics.accuracy_score(y_test, y_pred)
print("accuracy : {:2.3%}".format(accuracy))
# accuracy : 95.906%

# - of the predicted positives what percentage is truly positive
precision = metrics.precision_score(y_test, y_pred)
print("precision : {:2.3%}".format(precision))
# precision : 92.982%

# - how good is the model at predicting positives
sensitivity_recall = metrics.recall_score(y_test, y_pred)
print("sensitivity_recall : {:2.3%}".format(sensitivity_recall))
# sensitivity_recall : 94.643%

# - how good is the model at predicting negatives
specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
print("specificity : {:2.3%}".format(specificity))
# specificity : 96.522%

# - harmonic mean of precision and sensitivity_recall
f1_score = metrics.f1_score(y_test, y_pred)
print("f1_score : {:2.3%}".format(f1_score))
# f1_score : 93.805%

# finally show all the plots
# plt.show()
