from pervect.pervect_ import (
    gmm_component_likelihood,
    mat_sqrt,
    wasserstein2_gaussian,
    pairwise_gaussian_ground_distance,
    vectorize_diagram,
    pairwise_distances,
    GaussianMixture,
)

import pytest

import numpy as np

np.random.seed(42)
base_data = np.vstack(
    [
        np.random.beta(1, 5, size=100),
        np.random.gamma(shape=0.5, scale=1.0, size=100),
    ]
).T




def test_mat_sqrt():
    test_matrix = np.random.normal(size=(2, 2))
    # ensure it is symmetric
    test_matrix += test_matrix.T
    test_matrix = np.ascontiguousarray(test_matrix)

    root = mat_sqrt(test_matrix)
    square = root @ root

    assert np.allclose(test_matrix, square)


def test_gmm_component_likelihood():
    data = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
    result = gmm_component_likelihood(
        np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), data,
    )
    for i in range(data.shape[0]):
        likelihood = np.exp(0.5 * data[i] @ data[i]) / 2 * np.pi
        assert np.isclose(result[i], likelihood)


def test_wasserstein2_gaussian_simple_cov():

    means = np.random.random(size=(10, 2))
    covariance = np.array([[1, 0], [0, 1]])

    wass_dist = np.zeros((10, 10))
    for i in range(means.shape[0]):
        for j in range(means.shape[0]):
            wass_dist[i, j] = wasserstein2_gaussian(
                means[i], covariance, means[j], covariance,
            )

    euc_dist = pairwise_distances(means, metric="euclidean")

    assert np.allclose(wass_dist, euc_dist)


def test_pairwise_gaussian_ground_distance_simple_cov():

    means = np.random.random(size=(10, 2))
    covariances = np.dstack([np.array([[1, 0], [0, 1]]) for i in range(10)])

    wass_dist = pairwise_gaussian_ground_distance(means, covariances)
    euc_dist = pairwise_distances(means, metric="euclidean")

    assert np.allclose(wass_dist, euc_dist)


def test_vectorize_diagram():

    gmm = GaussianMixture(n_components=4, random_state=42).fit(base_data)
    result = vectorize_diagram(np.array([[0.5, 0.2],[0.75, 0.1]]), gmm)
    assert np.allclose(
        result, np.array(
            [6.24722853e-02, 5.50441490e-33, 0.00000000e+00, 1.93752771e+00]
        )
    )
