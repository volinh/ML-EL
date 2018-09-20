import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

np.random.seed(11)


def initial_data():
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[4, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis=0)
    K = 3
    original_label = np.asarray([0] * N + [1] * N + [2] * N).T
    return X,original_label,K


class Kmean():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


def kmean_sklearn(X,n_clusters):
    kmean = KMeans(n_clusters=n_clusters,)
    labels = kmean.fit_predict(X=X)
    return labels


def kmean_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=3, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=3, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=3, alpha=.8)

    plt.show()


if __name__ == "__main__" :
    X_data , true_labels , n_clusters = initial_data()
    pre_labes = kmean_sklearn(X_data,n_clusters)
    kmean_display(X_data,true_labels)
    kmean_display(X_data,pre_labes)






