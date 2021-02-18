#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)
# np.ndarray la N-dimensional array cua numpy
def plot_thresholds(x: np.ndarray, y: np.ndarray, thresholds: np.ndarray)->None:
    '''- Ham zip lai ghep cap no lai thanh tuple, cu moi tap hop boc tuong ung 1 thang tao thanh thang moi
       - Thuong dung de parallel loop.
       - [::-1] -> cai nay co nghia la lay tat ca cac hang cac cot roi lat nguoc lai.
       - Trong python co 2 loai tham so chua biet: 1 la *arg va **kwargs -> 2 thang nay deu dung khi khong
         biet nguoi dung se nhap bao nhieu tham so, *arg la nhan het tat ca value binh thuong, con **kwargs
         la nhan parameter voi 1 cap key = value -> nho day chuong trinh co the biet duoc la ban dang muon 
         nhap gi -> khi ma thay trong tham so co cac kieu a = b thi hieu do la kieu **kwargs'''
    
    for x, y, thresh in zip(x, y, thresholds[::-1]):
        plt.annotate(f"{thresh}", (x,y), textcoords="offset points", size=7, xytext=(-7, 7), ha='center')
        
        
def plot_pr_curve(y: np.ndarray, y_scores: np.ndarray):
    """Plots the precision recall curve given the true class labels y and the predicted labels with their probability"""
    # global thresholds
    # get precision recall values for all thresholds
    # Ham nay se nhan gia tri vao la gia tri y dung va array probability dung de du daon no
    # Ham se lien tuc tang muc nguong threshols chay dan tu thap len cao -> ve se ra ket qua
    precision, recall, thresholds = precision_recall_curve(
        y_true=y, probas_pred=y_scores[:, 1]
    )
    # Plot PR Curve
    plt.figure()
    plt.plot(precision, recall, "o-", label='Threshold')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plot_thresholds(precision, recall, thresholds)
    plt.legend(loc="lower left")
    plt.show()

    #raise NotImplementedError


def plot_roc_curve(y, y_scores):
    """Plots the Receiver Operating Characteristic Curve given the true class labels y and the predicted labels with
     their probability"""
    # Get tpr and fpr values for all thresholds
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores[:, 1])
    # Compute area under curve
    auc = roc_auc_score(y, y_scores[:, 1])
    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, 'o-', label='Threshold')
    plt.plot([0, 1], [0, 1], linestyle="--", c="r")
    plot_thresholds(fpr, tpr, thresholds)
    plt.title(f"ROC Curve ($AUC={auc:.2f}$)")
    plt.xlabel("False Positives Rate")
    plt.ylabel("True Positives Rate")
    plt.legend()
    plt.show()
    
    #raise NotImplementedError


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    """Plots the confusion matrix given the true test labels y_test and the predicted labes y_pred"""
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm)
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.grid(False)
    plt.show()
    #raise NotImplementedError


def main():
    # Load Data
    # Tu day co the thay trong nay co rat nhieu kieu du lieu -> doc them khi co thoi gian
    X, y = datasets.load_breast_cancer(return_X_y=True)
    print(X.shape) # (569,30)
    print(y.shape) # (569,)
    # Thang estimator la thang dung de phan nguong -> co thoi gian doc them thuat toan nay no nhu the nao
    clf = RandomForestClassifier()

    # Compute cross validation probabilities for each sample
    y_scores = cross_val_predict(
        estimator=clf, X=X, y=y, cv=5, n_jobs=-1, verbose=0, method="predict_proba"
    )
    
    print(y_scores)
    print(y_scores.shape) # (569, 2): tuong ung moi sample mot so xac suat xem no co doan trung khong
    
    plot_pr_curve(y, y_scores)
    plot_roc_curve(y, y_scores)
    
    print("I am here")
    # Load digits dataset
    X, y = datasets.load_digits(return_X_y=True)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plot_confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    main()
