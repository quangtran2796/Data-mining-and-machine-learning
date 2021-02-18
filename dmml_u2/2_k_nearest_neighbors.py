#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import utils


def load_reduced_iris_dataset() -> (np.ndarray, np.ndarray):
    """Loads the iris dataset reduced to its first two features (sepal width, sepal length)."""
    iris = datasets.load_iris()
    #Dau : dau tien la lay tat ca cac hang, :n la lay cac cot tu 0 -> n-1 
    X = iris.data[:, :2]
    y = iris.target
    
    return X,y
    raise NotImplementedError


def evaluate_ks(ks: range, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    """Evaluates each k-value of the ks-array and returns their respective training and test accuracy."""
    accuracies = np.array([utils.evaluate(k, X, y) for k in ks])
    acc_train = accuracies[:, 0]
    acc_test = accuracies[:, 1]
    return acc_train, acc_test
    raise NotImplementedError


def plot_k_to_acc(ks: range, acc_train: np.ndarray, acc_test: np.ndarray) -> None:
    """Plots the k-values in relation to their respective training and test accuracy."""
    #Ve thi cu moi lan dung lenh plot thi hinh do se duoc them vao trong do thi
    #Sau do neu muon them legend cho no thi goi ham plot.legend([them ten theo thu tu add vao trong hinh ngan cach boi dau ,])
    #Cach o tren thi ok, tuy nhien de sai sot vi phai nho thu tu -> Cach tot hon la them parameter label vao trong cac ham plot -> sau do goi plot.legend() no se tu dong them vao
    #Thay doi mau thi thay doi trong ham plot bang tham so color = ...
    #Thay doi kieu duong biey thi thi dung parameter linestyle = ...
    #Co the them cac marker qua tham so marker = ... Marker chinh la thang lam dau va lam noi len cac diem du lieu
    #Ham plt.scatter() cho phep ve cac diem duoi dang diem.
    
    plt.figure()
    plt.scatter(ks, acc_train, label="Accuracy Train", marker="x")
    plt.scatter(ks, acc_test, label="Accuracy Test", marker="^")
    plt.legend()
    plt.xlabel("$k$ Neighbors")
    plt.ylabel("Accuracy")
    plt.title("KNN: Accuracy vs. Number of Neighbors")
    plt.show()
    #raise NotImplementedError

def get_best_k(ks: range, acc_test: np.ndarray) -> int:
    """Returns the best value for k based on the highest test accuracy."""
    #Ham argmax() dung de tra ve index cua phan tu lon nhat.
    best_k = ks[acc_test.argmax()]
    return best_k
    raise NotImplementedError


def plot_decision_boundary_for_k(k: int, X: np.ndarray, y: np.ndarray) -> None:
    """Creates and fits a KNN model with value k and plots its decision boundary."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    utils.plot_decision_boundary(knn, X, y)
    #raise NotImplementedError


def main():
    # Set seed to make experiments reproducible
    np.random.seed(0)
    X, y = load_reduced_iris_dataset()
    utils.plot_iris_dataset(X, y)
    print(y)
    # Define interval
    ks = range(1, 100, 2)
    acc_train, acc_test = evaluate_ks(ks, X, y)

    plot_k_to_acc(ks, acc_train, acc_test)

    best_k = get_best_k(ks, acc_test)
    print(f"best k value: {best_k}")

    for k in range(1, 101, 20):
        plot_decision_boundary_for_k(k, X, y)


if __name__ == '__main__':
    main()
