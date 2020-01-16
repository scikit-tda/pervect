from pervect.pervect_ import (
    gmm_component_likelihood,
    mat_sqrt,
    wasserstein2_gaussian,
    pairwise_gaussian_ground_distance,
)

import pytest

import numpy as np

def test_mat_sqrt():
    test_matrix = np.random.random(2, 2)
    # ensure it is symmetric
    test_matrix += test_matrix.T
    test_matrix = np.ascontiguousarray(test_matrix)

    root = mat_sqrt(test_matrix)
    square = root @ root

    assert np.allclose(test_matrix, square)

def test_gmm_component_likelihood():
    data = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
    result = gmm_component_likelihood(
        np.array([0.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        data,
    )
    for i in range(data.shape[0]):
        likelihood = np.exp(0.5 * data[i] @ data[i]) / 2 * np.pi
        assert np.isclose(result[i], likelihood)
