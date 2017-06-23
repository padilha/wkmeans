import numpy as np

from scipy.spatial.distance import sqeuclidean
from sklearn.utils.validation import check_array

def wkmeans(data, k, weights, max_iter=500, tol=1e-4):
    data = check_array(data, dtype=np.double, copy=True)
    __check_params(data, k, weights, max_iter, tol)

    num_rows, num_cols = data.shape
    centroids = data[np.random.choice(num_rows, size=k, replace=False), :]

    for it in range(max_iter):
        old_centroids = np.copy(centroids)
        labels = __expectation(data, centroids)
        centroids = __maximization(data, weights, k, labels)

        if __sqfrobenius(centroids - old_centroids) <= tol:
            break

    return __expectation(data, centroids)

def __expectation(data, centroids):
    distances = np.array([[sqeuclidean(x, c) for c in centroids] for x in data])
    return np.argmin(distances, axis=1) # array of labels

def __maximization(data, weights, k, labels):
    return np.array([__update_centroid(data, weights, np.where(labels == i)[0]) for i in range(k)])

def __update_centroid(data, weights, indices):
    return np.sum(weights[indices, np.newaxis] * data[indices], axis=0) / np.sum(weights[indices])

def __sqfrobenius(x):
    return np.sum(x * x)

def __check_params(data, k, weights, max_iter, tol):
    if k <= 0 or k > data.shape[0]:
        raise ValueError("k must be > 0 and <= {}, got {}".format(data.shape[0], k))

    if len(weights) != data.shape[0]:
        raise ValueError("weights length expected {}, got {}".format(data.shape[0], len(weights)))

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0, got {}".format(max_iter))

    if tol < 0.0:
        raise ValueError("tol must be >= 0.0, got {}".format(tol))

if __name__ == '__main__':
    from sklearn import datasets
    data = datasets.load_iris().data
    print data.shape
    #weights = (np.arange(10) + 1) / 10.0
    weights = np.ones(data.shape[0], dtype=np.double)
    iris_weights = np.random.choice(weights, size=data.shape[0], replace=True)
    labels = wkmeans(data, k=3, weights=iris_weights, max_iter=500, tol=1e-4)
    print labels
