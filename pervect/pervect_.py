import numpy as np
import scipy.linalg
import scipy.stats
import numba
import ot
import umap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def wasserstein_diagram_distance(p, pts0, pts1, y_axis='death'):
    '''
    Compute the Persistant p-Wasserstein distance between the diagrams pts0, pts1
    
    y_axis = 'death' (default), or 'lifetime'
    
    '''
    
    if y_axis == 'lifetime':
        extra_dist0 = pts0[:, 1]
        extra_dist1 = pts1[:, 1]
    elif y_axis == 'death':    
        extra_dist0 = (pts0[:, 1]-pts0[:, 0])/np.sqrt(2)
        extra_dist1 = (pts1[:, 1]-pts1[:, 0])/np.sqrt(2)
    else:
        raise ValueError('y_axis must be \'death\' or \'lifetime\'')
        
    pairwise_dist = pairwise_distances(pts0, pts1)
    
    all_pairs_ground_distance_a = np.hstack([pairwise_dist, extra_dist0[:, np.newaxis]])
    extra_row = np.zeros(all_pairs_ground_distance_a.shape[1])
    extra_row[:pairwise_dist.shape[1]] = extra_dist1
    all_pairs_ground_distance_a = np.vstack([all_pairs_ground_distance_a, extra_row])
  
    all_pairs_ground_distance_a = all_pairs_ground_distance_a**p
    
    n0 = pts0.shape[0]
    n1 = pts1.shape[0]
    a = np.ones(n0+1)
    a[n0]=n1
    a = a/a.sum()
    b = np.ones(n1+1)
    b[n1]=n0
    b = b/b.sum()
    
    return np.power((n0+n1)*ot.emd2(a, b, all_pairs_ground_distance_a),1.0/p)


def gmm_component_likelihood(component_mean, component_covar, diagram):
    return scipy.stats.multivariate_normal.pdf(
        diagram,
        mean=component_mean,
        cov=component_covar,
    )


def vectorize_diagram(diagram, gmm):
    interim_matrix = np.zeros((gmm.n_components, diagram.shape[0]))
    for i in range(interim_matrix.shape[0]):
        interim_matrix[i] = gmm_component_likelihood(
            gmm.means_[i], gmm.covariances_[i], diagram
        )
    normalize(interim_matrix, norm='l1', axis=0, copy=False)
    return interim_matrix.sum(axis=1)


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


def add_birth_death_line(ground_distance, means, covariances, y_axis='death'):
    '''
    Return an appended ground distance matrix with the extra distance to the lifetime=0 line 
    '''
    
    if y_axis == 'lifetime':
        euclidean_dist = means[:, 1]
        anti_line = np.array([0,1])
    elif y_axis == 'death':    
        euclidean_dist = (means[:, 1]-means[:, 0])/np.sqrt(2)
        anti_line = 1/np.sqrt(2)*np.array([1,-1])
    else:
        raise ValueError('y_axis must be \'death\' or \'lifetime\'')
    
    variances = anti_line@covariances@anti_line.T     
    extra_dist = euclidean_dist + np.sqrt(variances)
    ground_distance_a = np.hstack([ground_distance, extra_dist[:, np.newaxis]])
    extra_row = np.zeros(ground_distance_a.shape[1])
    extra_row[:ground_distance.shape[1]] = extra_dist
    ground_distance_a = np.vstack([ground_distance_a, extra_row])
    return ground_distance_a
    

def persistence_wasserstein_distance(x, y, ground_distance):
    x_a = np.append(x, y.sum())
    x_a /= x_a.sum()
    y_a = np.append(y, x.sum())
    y_a /= y_a.sum()
    plan = ot.emd(x_a, y_a, ground_distance)
    return (x.sum() + y.sum()) * (plan * ground_distance).sum()


def persistence_p_wasserstein_distance(p, x, y, ground_distance):
    return np.power(persistence_wasserstein_distance(x,y,ground_distance**p),1/p)


class PersistenceVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=20, apply_umap=False, umap_n_components=2, y_axis='death'):
        self.n_components = n_components
        self.apply_umap = apply_umap
        self.umap_n_components = umap_n_components
        self.y_axis = y_axis

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
            self._raw_ground_distance, 
            self.mixture_model_.means_, 
            self.mixture_model_.covariances_,
            y_axis=self.y_axis
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
    
    
    def pairwise_p_wasserstein_distance(self, p, X):
        vecs = np.vstack(
            [
                vectorize_diagram(diagram, self.mixture_model_)
                for diagram in X
            ]
        )
        distance_matrix = pairwise_distances(
                vecs,
                metric=persistence_wasserstein_distance,
                ground_distance=self.ground_distance_**p
        )
        return np.power(distance_matrix, 1.0/p)

    

