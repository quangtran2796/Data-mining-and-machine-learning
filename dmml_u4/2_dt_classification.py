#!/usr/bin/env python
# coding: utf-8
# references:
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
# https://scikit-learn.org/stable/modules/tree.html

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
import utils


def get_unsure_test_items(clf, X_test: np.ndarray) -> np.ndarray:
    """Returns a boolean array where all items with a probability != 1 are marked as true"""
    #1 cai classifier bat ky thi se luon co 2 dang vorhersagen: 1 cai la predict nhu binh thuong
    #1 cai khac la predict_proba thi chi cho ra xac suat cua 1 ham la bao nhieu thoi.
    y_proba = clf.predict_proba(X_test)
    y_unsure = np.arange(len(X_test))
    
    for i in range(len(y_proba)):
        for j in y_proba[i,:]:
            if j == 1:
                y_unsure[i] = False
                break
            else:
                y_unsure[i] = True
    return y_unsure


def plot_decision_boundary(clf, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray, y_test_unsure: np.ndarray) -> None:
    """Create a decision boundary plot that shows the predicted label for each point."""
    
    cache = {}
    # Plot decision regions
    plt.figure()
    plt.title("Decision Tree Decision Regions on 2D Iris")
    h = 0.05
    # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    k = clf.min_samples_leaf
    #Cho cai dong if nay chua hieu -> coi video hoc lai.
    if k in cache:
        Z = cache[k]
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        cache[k] = Z
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training and test points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, marker="x", edgecolor="k", s
                        =25, label="Training Points")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, marker="^", edgecolor="k", s
                        =25, label="Test Points")
    plt.scatter(X_test[y_test_unsure][:, 0], X_test[y_test_unsure][:, 1], c="yellow", marker="*",
                edgecolor="k", s=50, label="Unsure Points")

    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def get_acc_per_max_depth(X: np.ndarray, y: np.ndarray, max_depths: range) -> [np.ndarray, np.ndarray]:
    """Runs 10-fold cross validation for all given depths and
     returns an array for the corresponding train and test accuracy"""
    
    def decision_tree_score(X, y, md):
        
        my_model = tree.DecisionTreeClassifier(max_depth = md)
        scores = cross_validate(
         my_model,
         X = X,
         y = y,
         cv = 10,
         n_jobs=-1,
         scoring = 'accuracy',
         return_train_score = True)
         
        return np.mean(scores["test_score"]), np.mean(scores["train_score"])
    
    accuracies = np.array([decision_tree_score(X, y, md) for md in max_depths])
    acc_test = accuracies[:,0]
    acc_train = accuracies[:,1]

    return acc_train, acc_test


def main():
    # Load the iris dataset
    iris = load_iris()

    # Fit a Decision tree on the iris data
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    utils.plot_decision_tree(clf, iris)

    X, y = utils.reduce_iris_to_2d(iris)

    # Train decision tree
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = clf.fit(X_train, y_train)

    y_test_unsure = get_unsure_test_items(clf, X_test)

    plot_decision_boundary(clf, X_train, X_test, y_train, y_test, y_test_unsure)

    # Make experiments reproducible
    np.random.seed(0)

    # Define interval
    max_depths = range(1, 16)
    acc_train, acc_test = get_acc_per_max_depth(X, y, max_depths)

    utils.plot_results(acc_test, acc_train, max_depths)


if __name__ == '__main__':
    main()
