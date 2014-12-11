#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score

# Read the training X and y
reader = csv.reader(open("../data/train.csv"))
Xtrain = np.asarray([line for line in reader], dtype = np.float64)
reader = csv.reader(open("../data/trainLabels.csv"))
ytrain = np.asarray([line[0] for line in reader], dtype = np.int)

# 0. Simple statistics
sum(ytrain) # Number of positive samples. Skewed?
Xtrain.mean(axis=0) # Mean of each feature
Xtrain.std(axis=0) # Std of each feature
Xtrain_scaled = preprocessing.scale(Xtrain) # Scale the data

# 0. PCA for visualization
from sklearn.decomposition import PCA

pca = PCA()
Xtrain_r = pca.fit(Xtrain).transform(Xtrain)
Xtrain_scaled_r = pca.fit(Xtrain_scaled).transform(Xtrain_scaled)
plt.subplot(1, 2, 1)
for c, i in zip("rg", [0, 1]):
    plt.scatter(Xtrain_r[ytrain == i, 0], Xtrain_r[ytrain == i, 1], c=c, label=str(i))
plt.legend()
plt.title("PCA of unscaled dataset")
plt.subplot(1, 2, 2)
for c, i in zip("rg", [0, 1]):
    plt.scatter(Xtrain_scaled_r[ytrain == i, 0], Xtrain_scaled_r[ytrain == i, 1], c=c, label=str(i))
plt.legend()
plt.title("PCA of scaled dataset")
plt.show()

# 1. Logistic regression as baseline
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
# CV
logistic_scores = cross_val_score(logistic, Xtrain_scaled, ytrain, cv=10)
logistic_scores.mean()
# Add L1 penalty
for C in (10.0 ** np.arange(-1.6, -0.2, 0.2)):
    logistic_l1 = LogisticRegression(C=C, penalty='l1')
    logistic_l1.fit(Xtrain_scaled, ytrain)
    
    logistic_l1_coef = logistic_l1.coef_.ravel()
    logistic_l1_sparsity = np.mean(logistic_l1_coef == 0) * 100
    print("C=%.2f" % C)
    print("Sparsity with L1 penalty: %.2f%%" % logistic_l1_sparsity)
    logistic_l1_scores = cross_val_score(logistic_l1, Xtrain_scaled, ytrain, cv=10)
    print("CV score with L1 penalty: %.4f" % logistic_l1_scores.mean())
    print

# Decision boundary is non-linear?

# 2. Random forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 200, random_state=0) # There are many parameters to play with here, such as n_estimators, max_features, etc.
forest.fit(Xtrain, ytrain)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1][0:10]
# Print the feature ranking
print("Feature ranking:")
for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()
# CV
forest_scores = cross_val_score(forest, Xtrain, ytrain, cv=10)
forest_scores.mean()
