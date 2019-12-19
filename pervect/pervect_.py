import numpy as np
import scipy.linalg
import scipy.stats
import numba
import ot
import umap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

def gmm_component_likelihood(component_mean, component_covar, diagram):
    return scipy.stats.multivariate_normal.pdf(
        diagram,
        mean=component_mean,
        cov=component_covar,
    ).sum()


def vectorize_diagram(diagram, gmm):
    result = np.zeros(gmm.n_components)
    for i in range(result.shape[0]):
        result[i] = gmm_component_likelihood(
            gmm.means_[i], gmm.covariances_[i], diagram
        )
    return result


@numba.njit()
def mat_sqrt(mat):
    result = mat.copy()
    s = mat[0,0] * mat[1,1] - mat[1,0] * mat[0,1]
    factor = 1.0 / (mat[0,0] + mat [1,1] + 2.0 * s)
    result[0,0] = result[0,0] * np.sqrt(factor) + np.sqrt(s * factor)
    result[1,1] = result[1,1] * np.sqrt(factor) + np.sqrt(s * factor)
    result[0,1] *= np.sqrt(factor)
    result[1,0] *= np.sqrt(factor)
    return result


@numba.njit()
def wasserstein2_gaussian(m1, C1, m2, C2):
    result = np.sum((m1 - m2)**2)
    sqrt_C2 = mat_sqrt(C2)
    prod_matrix = sqrt_C2 @ C1 @ sqrt_C2
    sqrt_prod_matrix = mat_sqrt(prod_matrix)
    correction_matrix = C1 + C2 - 2 * sqrt_prod_matrix
    result += correction_matrix[0,0] + correction_matrix[1,1]
    return np.sqrt(np.maximum(result, 0))


@numba.njit()
def pairwise_gaussian_ground_distance(means, covariances):
    n_components = means.shape[0]

    result = np.zeros((n_components, n_components), dtype=np.float32)
    for i in range(n_components):
        for j in range(i + 1, n_components):
            result[i, j] = wasserstein2_gaussian(
                means[i], covariances[i], means[j], covariances[j]
            )
            result[j, i] = result[i, j]

    return result


def add_birth_death_line(ground_distance, means):
    extra_column = means[:, 1].astype(np.float32) # lifetime of component
    result = np.hstack([ground_distance, extra_column[:, np.newaxis]])
    extra_row = np.zeros(result.shape[1], dtype=np.float32)
    extra_row[:ground_distance.shape[0]] = means[:, 1].astype(np.float32)
    result = np.vstack([result, extra_row])
    return result


def persistence_wasserstein_distance(x, y, ground_distance):
    x_a = np.append(x, y.sum())
    x_a /= x_a.sum()
    y_a = np.append(y, x.sum())
    y_a /= y_a.sum()
    plan = ot.emd(x_a, y_a, ground_distance)
    return np.sqrt((x.sum() + y.sum()) * (plan * ground_distance).sum())


class PersistenceVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=20, apply_umap=False, umap_n_components=2):
        self.n_components = n_components
        self.apply_umap = apply_umap
        self.umap_n_components = umap_n_components


    def fit(self, X):
        # TODO: verify we have diagrams of the appropriate form
        diagram_union = np.vstack(X)
        self.mixture_model_ = GaussianMixture(n_components=self.n_components)
        self.mixture_model_.fit(diagram_union)
        self._raw_ground_distance = pairwise_gaussian_ground_distance(
            self.mixture_model_.means_,
            self.mixture_model_.covariances_,
        )
        self.ground_distance_ = add_birth_death_line(
            self._raw_ground_distance, self.mixture_model_.means
        )

        return self

    def transform(self, X, y=None):
        result = np.vstack(
            [
                vectorize_diagram(diagram, self.mixture_model_)
                for diagram in X
            ]
        )

        if self.apply_umap:
            distance_matrix = pairwise_distances(
                result,
                metric=persistence_wasserstein_distance,
                ground_distance=self.ground_distance_,
            )
            result = umap.UMAP(
                metric="precomputed",
                n_components=self.umap_n_components,
            ).fit_transform(distance_matrix)

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)