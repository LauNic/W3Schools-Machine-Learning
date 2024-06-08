# Decision Trees
# - are flow charts that can make decisions based on prev experience

# Problem: a person should decide if he will go to a comedy or not
# - we have a dataset with info about the comedian
# -- based on the data Python will create a decision tree to decide to go or no

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

comedy_data_frame = pd.read_csv("./files/comedy_data.csv", sep=";")
print(comedy_data_frame)
print(comedy_data_frame.describe())
print(comedy_data_frame.groupby(["Nationality"]).count().sort_values(by="Nationality", ascending=False))
print(comedy_data_frame.groupby(["Go"]).count().sort_values(by="Go", ascending=False))

# Decision trees needs that all data is numerical
nationality_dictionary = {"USA": 1, "UK": 0, "N": 2}
go_dictionary = {"YES": 1, "NO": 0}
# use the pandas map() method to transform value into numerical
comedy_data_frame["Nationality"] = comedy_data_frame["Nationality"].map(nationality_dictionary)
comedy_data_frame["Go"] = comedy_data_frame["Go"].map(go_dictionary)
print(comedy_data_frame)

# separate  feature columns from the target column "Go"
feature_columns = comedy_data_frame.keys().drop(["Go"]).values
train_X = comedy_data_frame[feature_columns]
print(train_X)
train_y = comedy_data_frame["Go"]
print(train_y)

# build a decision tree classifier from the training set
decision_tree = DecisionTreeClassifier().fit(train_X.values, train_y.values)
# predict if Go to comedy using the Decision Tree
predicted_y_1 = decision_tree.predict([[40, 10, 7, 1]])
key_dictionary_predicted_1 = list(go_dictionary.keys())[list(go_dictionary.values()).index(predicted_y_1)]
print("Prediction [40, 10, 7, 1] :", key_dictionary_predicted_1)

predicted_y_2 = decision_tree.predict([[40, 10, 6, 1]])
key_dictionary_predicted_2 = list(go_dictionary.keys())[list(go_dictionary.values()).index(predicted_y_2)]
print("Prediction [40, 10, 6, 1] :", key_dictionary_predicted_2)

# plot the Decision Tree flow chart
tree.plot_tree(decision_tree, feature_names=feature_columns)
plt.show()
