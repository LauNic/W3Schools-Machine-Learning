# Cross Validation
# -- is used for increasing overall performance on unseen data 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (KFold, cross_val_score, StratifiedKFold,
                                     LeaveOneOut, LeavePOut, ShuffleSplit)

X, y = load_iris(return_X_y=True)

# K-Fold
# - data is split into k-sets
# -- model is trained on k-1 folds
# --- 1 remaining Fold is used as validation set to evaluate the model
decision_tree = DecisionTreeClassifier(random_state=42)
k_folds = KFold(n_splits=5)
scores = cross_val_score(decision_tree, X, y, cv=k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


decision_tree2 = DecisionTreeClassifier(random_state=42)
sk_folds = StratifiedKFold(n_splits=5)
scores2 = cross_val_score(decision_tree2, X, y, cv=sk_folds)

print("Cross Validation Scores: ", scores2)
print("Average CV Score: ", scores2.mean())
print("Number of CV Scores used in Average: ", len(scores2))


decision_tree3 = DecisionTreeClassifier(random_state=42)
leave_1o = LeaveOneOut()
scores3 = cross_val_score(decision_tree3, X, y, cv=leave_1o)

print("Cross Validation Scores: ", scores3)
print("Average CV Score: ", scores3.mean())
print("Number of CV Scores used in Average: ", len(scores3))

decision_tree4 = DecisionTreeClassifier(random_state=42)
leave_p_o = LeavePOut(p=2)
scores4 = cross_val_score(decision_tree4, X, y, cv=leave_p_o)

print("Cross Validation Scores: ", scores4)
print("Average CV Score: ", scores4.mean())
print("Number of CV Scores used in Average: ", len(scores4))

decision_tree5 = DecisionTreeClassifier(random_state=42)
shuffle_split = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores5 = cross_val_score(decision_tree5, X, y, cv=shuffle_split)

print("Cross Validation Scores: ", scores5)
print("Average CV Score: ", scores5.mean())
print("Number of CV Scores used in Average: ", len(scores5))
