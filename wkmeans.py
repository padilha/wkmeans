"""
    wkmeans: a simple implementation of the k-means clustering algorithm with weighted objects.
    Copyright (C) 2017  Victor Alexandre Padilha

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from scipy.spatial.distance import sqeuclidean
from sklearn.utils.validation import check_array

def wkmeans(data, k, weights, max_iter=500, tol=1e-4):
    """Performs weighted k-means.

    Parameters
    ----------
    data : numpy.ndarray (n_objects, n_features)
        Dataset to cluster.

    k : int
        Number of clusters to form.

    weights : numpy.array (n_objects)
        Weights of the objects contained in the dataset.

    max_iter : int
        Maximum number of iterations to perform.

    tol : float
        Maximum total shift of the centroids between two iterations (measured by the squared Frobenius norm).
    """
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
