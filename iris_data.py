## Imports
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

from sklearn import datasets

## Load Data
print("IRIS DATA")
print()

iris = datasets.load_iris()

## Train Models

accs, recs, precs, f1s = [], [], [], []
clfs = []

for i in range(1, 11):
    print("עומק ", i)
    print()
    clf = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy')
    clf = clf.fit(iris.data, iris.target)
    clfs.append(clf)

    accuracy = cross_val_score(clf, iris.data, iris.target, scoring='accuracy', cv=10)
    accs.append(accuracy.mean())

    recall = cross_val_score(clf, iris.data, iris.target, scoring='recall_weighted', cv=10)
    recs.append(recall.mean())

    precision = cross_val_score(clf, iris.data, iris.target, scoring='precision_weighted', cv=10)
    precs.append(precision.mean())

    f1 = cross_val_score(clf, iris.data, iris.target, scoring='f1_weighted', cv=10)
    f1s.append(f1.mean())
    print()
    print("Average Accuracy of DT with depth: ", max(accs), "\t | depth:", np.argmax(accs) + 1)
    print("Average Recall of DT with depth ", max(recs), "\t | depth:", np.argmax(recs) + 1)
    print("Average precision_weighted of DT with depth ", max(precs), "\t | depth:", np.argmax(precs) + 1)
    print("Average f1_weighted of DT with depth ", max(f1s), "\t | depth:", np.argmax(f1s) + 1)
    print()
    plt.figure(figsize=(6, 6))

    plt.plot(list(range(1, len(accs) + 1)), accs, color="b", label="Accuracy")
    plt.plot(list(range(1, len(recs) + 1)), recs, color="y", label="Recall")
    plt.plot(list(range(1, len(precs) + 1)), precs, color="g", label="Precision")
    plt.plot(list(range(1, len(f1s) + 1)), f1s, color="r", label="F1")

    plt.title("Evolution of Accuracy, Precision and F1 Score with max depth")

    plt.xlabel("Max depth")
    plt.ylabel("Metric")
    plt.legend(loc="upper right")

    plt.show()
