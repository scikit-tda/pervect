from pervect.pervect_ import (
    gmm_component_likelihood,
    mat_sqrt,
    wasserstein2_gaussian,
    pairwise_gaussian_ground_distance,
    vectorize_diagram,
    add_birth_death_line,
    wasserstein_diagram_distance,
    persistence_p_wasserstein_distance,
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

    np.random.seed(42)
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
        likelihood = np.exp(-0.5 * (data[i] @ data[i])) / (2 * np.pi)
        assert np.isclose(result[i], likelihood)


def test_wasserstein2_gaussian_simple_cov():

    means = np.random.random(size=(10, 2))
    covariance = np.array([[1.0, 0.0], [0.0, 1.0]])

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
    covariances = np.dstack([np.array([[1.0, 0.0], [0.0, 1.0]]) for i in range(10)]).T

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


def test_add_birth_death_line():

    means = np.random.random(size=(10, 2))
    covariances = np.dstack([np.array([[1.0, 0.0], [0.0, 1.0]]) for i in range(10)]).T
    ground_dist = pairwise_gaussian_ground_distance(means, covariances)

    new_ground_dist = add_birth_death_line(ground_dist, means, covariances, y_axis="lifetime")
    assert np.allclose(new_ground_dist[-1, :-1], means[:, 1] + 1.0)

    new_ground_dist = add_birth_death_line(ground_dist, means, covariances, y_axis="death")
    assert np.allclose(new_ground_dist[-1, :-1],
                       ((means[:, 1] - means[:,0]) / np.sqrt(2)) + 1.0)


def test_wasserstein_diagram_distance():

    pts0 = base_data[:10]
    pts1 = base_data[-10:]

    d_death_1 = wasserstein_diagram_distance(pts0, pts1, "death", 1)
    assert np.isclose(d_death_1, 0.6218)

    d_life_1 = wasserstein_diagram_distance(pts0, pts1, "lifetime", 1)
    assert np.isclose(d_life_1, 1.17625)


def test_persistence_p_wasserstein_distance():

    gmm = GaussianMixture(n_components=4, random_state=42).fit(base_data[10:90])
    v1 = vectorize_diagram(base_data[:10], gmm)
    v2 = vectorize_diagram(base_data[-10:], gmm)

    raw_ground_distance = pairwise_gaussian_ground_distance(
        gmm.means_, gmm.covariances_,
    )
    ground_distance = add_birth_death_line(
        raw_ground_distance,
        gmm.means_,
        gmm.covariances_,
        y_axis="lifetime",
    )

    d = persistence_p_wasserstein_distance(v1, v2, ground_distance)

    assert np.isclose(d, 0.94303)


