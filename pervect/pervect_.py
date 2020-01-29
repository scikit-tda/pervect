import numba
import numpy as np
import ot
import scipy.linalg
import scipy.stats
import umap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state
from typing import Union, Sequence, AnyStr

from warnings import warn


def wasserstein_diagram_distance(
    pts0: np.ndarray, pts1: np.ndarray, y_axis: AnyStr = "death", p: int = 1
) -> float:
    """Compute the Persistant p-Wasserstein distance between the diagrams pts0, pts1

    Parameters
    ----------
    pts0: array of shape (n_top_features, 2)
        The first persistence diagram

    pts1: array of shape (n_top_features, 2)
        Thew second persistence diagram

    y_axis: optional, default="death"
        What the y-axis of the diagram represents. Should be one of
            * ``"lifetime"``
            * ``"death"``

    p: int, optional (default=1)
        The p in the p-Wasserstein distance to compute

    Returns
    -------
    distance: float
        The p-Wasserstein distance between diagrams ``pts0`` and ``pts1``
    """
    if y_axis == "lifetime":
        extra_dist0 = pts0[:, 1]
        extra_dist1 = pts1[:, 1]
    elif y_axis == "death":
        extra_dist0 = (pts0[:, 1] - pts0[:, 0]) / np.sqrt(2)
        extra_dist1 = (pts1[:, 1] - pts1[:, 0]) / np.sqrt(2)
    else:
        raise ValueError("y_axis must be 'death' or 'lifetime'")

    pairwise_dist = pairwise_distances(pts0, pts1)

    all_pairs_ground_distance_a = np.hstack([pairwise_dist, extra_dist0[:, np.newaxis]])
    extra_row = np.zeros(all_pairs_ground_distance_a.shape[1])
    extra_row[: pairwise_dist.shape[1]] = extra_dist1
    all_pairs_ground_distance_a = np.vstack([all_pairs_ground_distance_a, extra_row])

    all_pairs_ground_distance_a = all_pairs_ground_distance_a ** p

    n0 = pts0.shape[0]
    n1 = pts1.shape[0]
    a = np.ones(n0 + 1)
    a[n0] = n1
    a = a / a.sum()
    b = np.ones(n1 + 1)
    b[n1] = n0
    b = b / b.sum()

    return np.power((n0 + n1) * ot.emd2(a, b, all_pairs_ground_distance_a), 1.0 / p)


def gmm_component_likelihood(
    component_mean: np.ndarray, component_covar: np.ndarray, diagram: np.ndarray
) -> np.ndarray:
    """Generate the vector of likelihoods of observing points in a diagram
    for a single gmm components (i.e. a single Gaussian). That is, evaluate
    the given Gaussian PDF at all points in the diagram.

    Parameters
    ----------
    component_mean: array of shape (2,)
        The mean of the Gaussian

    component_covar: array of shape (2,2)
        The covariance matrix of the Gaussian

    diagram: array of shape (n_top_features, 2)
        The persistence diagram

    Returns
    -------
    likelihoods: array of shape (n_top_features,)
        The likelihood of observing each topological feature in the diagram
        under the provided Gaussian
    """
    return scipy.stats.multivariate_normal.pdf(
        diagram, mean=component_mean, cov=component_covar,
    )


def vectorize_diagram(diagram: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """Given a diagram and a Guassian Mixture Model, produce the vectorized
    representation of the diagram as a vector of weights associated to
    each component of the GMM.

    Parameters
    ----------
    diagram: array of shape (n_top_features, 2)
        The persistence diagram to be vectorized

    gmm: sklearn.mixture.GaussianMixture
        The Gaussian Mixture Model to use for vectorization

    Returns
    -------
    vect: array of shape (gmm.n_components,)
        The vector representation of the persistence diagram
    """
    interim_matrix = np.zeros((gmm.n_components, diagram.shape[0]))
    for i in range(interim_matrix.shape[0]):
        interim_matrix[i] = gmm_component_likelihood(
            gmm.means_[i], gmm.covariances_[i], diagram
        )
    normalize(interim_matrix, norm="l1", axis=0, copy=False)
    return interim_matrix.sum(axis=1)


@numba.njit()
def mat_sqrt(mat: np.ndarray) -> np.ndarray:
    """Closed form solution for the square root of a 2x2 matrix

    Parameters
    ----------
    mat: array of shape (2,2)
        The matrix to take the square root of

    Returns
    -------
    root: array of shape (2,2)
        The matrix such that root * root == mat (up to precision)
    """
    result = mat.copy()
    s = np.sqrt(mat[0, 0] * mat[1, 1] - mat[1, 0] * mat[0, 1])
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2.0 * s)
    result[0, 0] += s
    result[1, 1] += s
    result /= t
    return result


@numba.njit()
def wasserstein2_gaussian(
    m1: np.ndarray, C1: np.ndarray, m2: np.ndarray, C2: np.ndarray
) -> float:
    """Compute the Wasserstein_2 distance between two 2D Gaussians. This can be
    computed via the closed form formula:

    $$W_{2} (\mu_1, \mu_2)^2 = \| m_1 - m_2 \|_2^2 + \mathop{\mathrm{trace}} \bigl( C_1 + C_2 - 2 \bigl( C_2^{1/2} C_1 C_2^{1/2} \bigr)^{1/2} \bigr)$$

    Parameters
    ----------
    m1: array of shape (2,)
        Mean of the first Gaussian

    C1: array of shape (2,2)
        Covariance matrix of the first Gaussian

    m1: array of shape (2,)
        Mean of the second Gaussian

    C2: array of shape (2,2)
        Covariance matrix of the second Gaussian

    Returns
    -------
    dist: float
        The Wasserstein_2 distance between the two Gaussians
    """
    result = np.sum((m1 - m2) ** 2)
    sqrt_C2 = np.ascontiguousarray(mat_sqrt(C2))
    prod_matrix = sqrt_C2 @ C1 @ sqrt_C2
    sqrt_prod_matrix = mat_sqrt(prod_matrix)
    correction_matrix = C1 + C2 - 2 * sqrt_prod_matrix
    result += correction_matrix[0, 0] + correction_matrix[1, 1]
    return np.sqrt(np.maximum(result, 0))


@numba.njit()
def pairwise_gaussian_ground_distance(
    means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Compute pairwise distances between a list of Gaussians. This can be
    used as the ground distance for an earth-mover distance computation on
    vectorized persistence diagrams.

    Parameters
    ----------
    means: array of shape (n_gaussians, 2)
        The means for the Gaussians

    covariances: array of shape (n_gaussians, 2, 2)
        The covariance matrrices of the Gaussians

    Returns
    -------
    dist_matrix: array of shape (n_gaussians, n_gaussians)
        The pairwise Wasserstein_2 distance between the Gaussians
    """
    n_components = means.shape[0]

    result = np.zeros((n_components, n_components), dtype=np.float32)
    for i in range(n_components):
        for j in range(i + 1, n_components):
            result[i, j] = wasserstein2_gaussian(
                means[i], covariances[i], means[j], covariances[j]
            )
            result[j, i] = result[i, j]

    return result


def add_birth_death_line(
    ground_distance: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    y_axis: AnyStr = "death",
) -> np.ndarray:
    """Return an appended ground distance matrix with the extra distance to
    the lifetime=0 line. This provides a ground-distance for points in a
    persistence diagram to be removed via moving them to the line provided
    by lifetime=0, making this a true persistence diagram wasserstein
    distance approximation when computed as an earth-mover distance under this
    ground-distance.

    Parameters
    ----------
    ground_distance: array of shape (n_gaussians, n_gaussians)
        The current ground distance matrix of pairwise Wasserstein_2 distance
        between the Gaussians.

    means: array of shape (n_gaussians, 2)
       The means for the Gaussians

    covariances: array of shape (n_gaussians, 2, 2)
       The covariance matrrices of the Gaussians

    y_axis: optional, (default="death")
        What the y-axis of the diagram represents. Should be one of
            * ``"lifetime"``
            * ``"death"``

    Returns
    -------
    new_ground_distance: array of shape (n_gaussians + 1, n_gaussians + 1)
        The amended matrix to be used as a ground-distance matrix
    """
    if y_axis == "lifetime":
        euclidean_dist = means[:, 1]
        anti_line = np.array([0, 1])
    elif y_axis == "death":
        euclidean_dist = (means[:, 1] - means[:, 0]) / np.sqrt(2)
        anti_line = 1 / np.sqrt(2) * np.array([1, -1])
    else:
        raise ValueError("y_axis must be 'death' or 'lifetime'")

    variances = anti_line @ covariances @ anti_line.T
    extra_dist = euclidean_dist + np.sqrt(variances)
    ground_distance_a = np.hstack([ground_distance, extra_dist[:, np.newaxis]])
    extra_row = np.zeros(ground_distance_a.shape[1])
    extra_row[: ground_distance.shape[1]] = extra_dist
    ground_distance_a = np.vstack([ground_distance_a, extra_row])
    return ground_distance_a


def persistence_wasserstein_distance(
    x: np.ndarray, y: np.ndarray, ground_distance: np.ndarray
) -> float:
    """Compute an approximation of Persistence Wasserstein_1 distance
    between persistenced iagrams with vector representations ``x`` and ``y``
    using the ground distance provided.

    Parameters
    ----------
    x: array of shape (n_gaussians,)
        The vectorization of the first persistence diagram

    y: array of shape (n_gaussians,)
        The vectorization of the first persistence diagram

    ground_distance: array of shape (n_gaussians + 1, n_gaussians + 1)
        The amended ground-distance as output by ``add_birth_death_line``

    Returns
    -------
    dist: float
        Ann approximation of Persistence Wasserstein_1 distance
        between persistenced iagrams with vector representations
        ``x`` and ``y``
    """
    x_a = np.append(x, y.sum())
    x_a /= x_a.sum()
    y_a = np.append(y, x.sum())
    y_a /= y_a.sum()
    plan = ot.emd(x_a, y_a, ground_distance)
    return (x.sum() + y.sum()) * (plan * ground_distance).sum()


def persistence_p_wasserstein_distance(
    x: np.ndarray, y: np.ndarray, ground_distance: np.ndarray, p: int = 1
) -> float:
    """Compute an approximation of Persistence Wasserstein_p distance
    between persistenced diagrams with vector representations ``x`` and ``y``
    using the ground distance provided, and p=``p``.

    Parameters
    ----------
    x: array of shape (n_gaussians,)
        The vectorization of the first persistence diagram

    y: array of shape (n_gaussians,)
        The vectorization of the first persistence diagram

    ground_distance: array of shape (n_gaussians + 1, n_gaussians + 1)
        The amended ground-distance as output by ``add_birth_death_line``

    p: int, optional (default=1)
        The p in the p-Wasserstein distance to compute

    Returns
    -------
    dist: float
        Ann approximation of Persistence Wasserstein_1 distance
        between persistenced iagrams with vector representations
        ``x`` and ``y``
    """
    return np.power(persistence_wasserstein_distance(x, y, ground_distance ** p), 1 / p)


class PersistenceVectorizer(BaseEstimator, TransformerMixin):

    """Vectorizer for persistence diagrams. Given a training set set of persistence
    diagrams this class will fit a Gaussian mixture model to the set of all points
    across all diagrams and vectorize a diagram as the MLE of mix weights for
    the existing components given the diagram. This provides a very low-dimensional
    but informative vectorization fo the data. Moreover this vectorization can be
    used to compute approximations to p-Wassertstein distance between diagrams.

    A potential further vectorization step, using UMAP to convert p-Wassertstein
    distance approxmations to a low dimensional euclidean representation can also
    be applied, allowing easy visualization of the persistence diagram space
    itself, even under a suitable metric. This option can be enabled with
    the ``apply_umap`` parameter.

    Parameters
    ----------
    n_components: int (optional, default=20)
        The number of components or dimensions to use in the vectorized representation.

    apply_umap: bool (optional, default=False)
        Whether to apply UMAP to the results to generate a low dimensional euclidean
        space representation of the diagrams.

    umap_n_components: int (optional, default=2)
        The number of dimensions of euclidean space to use when representing the
        diagrams via UMAP.

    umap_metric: string (optional, default="hellinger")
        What metric to use for the UMAP embedding if ``apply_umap`` is enabled (
        this option will be ignored if ``apply_umap`` is ``False``). Should be one
        of:
            * ``"wasserstein"``
            * ``"hellinger"``
        Note that if ``"wasserstein"`` is used then transforming new data
        will not be possible.

    p: int (optional, default=1)
        The default p value to use when computing p-Wasserstein distance

    y_axis: string (optional, default="death")
        What the y-axis represents in the diagrams. The options are either
        birth-death diagrams or birth-lifetime diagrams. This parameter should
        be one of:
            * ``"death"``
            * ``"lifetime"``

    Attributes
    ----------
    mixture_model_: sklearn.mixture.GaussianMixture
        The Gaussian mixture model that was fit to the complete set of training
        diagrams.

    ground_distance_: np.array(n_components + 1, n_components + 1)
        The all pairs distance matrix of Wasserstein distance between the components
        of the mixture model, plus an extra component representing the zero-lifetime
        line. This, in turn, can be used as a means to compute Wassersteing distance
        between vectorizations by computing an earth-mover distance with this matrix
        as the ground-distance or cost matrix.
    """

    def __init__(
        self,
        n_components=20,
        apply_umap=False,
        umap_n_components=2,
        umap_metric="hellinger",
        p=1,
        y_axis="death",
        random_state=None,
    ):
        self.n_components = n_components
        self.apply_umap = apply_umap
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.y_axis = y_axis
        self.p = p
        self.random_state = random_state

    def _validate_params(self):
        if (
            not np.issubdtype(type(self.n_components), np.integer)
            or self.n_components < 2
        ):
            raise ValueError(
                "n_components must be and integer greater than or equal " "to 2."
            )
        if (
            not np.issubdtype(type(self.umap_n_components), np.integer)
            or self.umap_n_components < 2
        ):
            raise ValueError(
                "umap_n_components must be and integer greater than or " "equal to 2."
            )
        if (
            self.umap_n_components != 2 or self.umap_metric != "hellinger"
        ) and self.apply_umap is False:
            warn(
                "apply_umap was False, so umap_n_components and umap_metric will be "
                "ignored! Did you mean to set apply_umap=True?"
            )

    def fit(self, X: Sequence[np.ndarray], y=None):
        """Fit a pervect model to the list of persistence diagrams X

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : list or tuple of arrays of shape (N, 2)
            The diagrams to fit the model to. Each diagram should be an array
            of shape (N, 2) for varying N, where each row of the array is the
            birth-death or birth-lifetime coordinates of a topological feature
            in the diagram. ``X`` should then be a list or tuple of such arrays.
        """
        random_state = check_random_state(self.random_state)
        self._validate_params()

        try:
            diagram_union = np.vstack(X)
        except:
            raise ValueError(
                "Input data is not a list or tuple of diagrams! "
                "Please provide a list of ndarrays of diagrams."
            )

        diagram_union = check_array(diagram_union)
        if diagram_union.shape[1] != 2:
            raise ValueError(
                "Input data is not a list or tuple of diagrams! "
                "Please provide a list of ndarrays of diagrams. "
                "Each diagram should be an array of points in the "
                "plane corresponding to either the birth-death "
                "or the birth-lifetime coordinates of an observed "
                "topological feature."
            )

        self.mixture_model_ = GaussianMixture(
            n_components=self.n_components, random_state=random_state
        )
        self.mixture_model_.fit(diagram_union)
        self._raw_ground_distance = pairwise_gaussian_ground_distance(
            self.mixture_model_.means_, self.mixture_model_.covariances_,
        )
        self.ground_distance_ = add_birth_death_line(
            self._raw_ground_distance,
            self.mixture_model_.means_,
            self.mixture_model_.covariances_,
            y_axis=self.y_axis,
        )

        if self.apply_umap:
            random_state = check_random_state(self.random_state)
            self.train_vectors_ = np.vstack(
                [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
            )
            if self.umap_metric == "wasserstein":
                self._distance_matrix = pairwise_distances(
                    self.train_vectors_,
                    metric=persistence_wasserstein_distance,
                    ground_distance=self.ground_distance_ ** self.p,
                )
                self._distance_matrix = np.power(self._distance_matrix, 1.0 / self.p)
                random_state = check_random_state(self.random_state)
                self.umap_ = umap.UMAP(
                    metric="precomputed",
                    n_components=self.umap_n_components,
                    random_state=random_state,
                ).fit(self._distance_matrix)
            elif self.umap_metric == "hellinger":
                try:
                    self.umap_ = umap.UMAP(
                        metric="hellinger",
                        n_components=self.umap_n_components,
                        random_state=random_state,
                    ).fit(self.train_vectors_)
                except ValueError:
                    warn("Hellinger metric not available ... falling back to cosine")
                    self.umap_ = umap.UMAP(
                        metric="cosine",
                        n_components=self.umap_n_components,
                        random_state=random_state,
                    ).fit(self.train_vectors_)
            else:
                raise ValueError(
                    'umap_metric shoud be one of "wasserstein" or ' '"hellinger".'
                )

        return self

    def transform(self, X: Sequence[np.ndarray], y=None) -> np.ndarray:
        """Transform a list or tuple of persistence diagrams into a pervect
        representation. This will provide a vector representation of the diagrams
        provided by X as an array with one row for each diagram in the same order
        as the diagrams were passed in.

        Parameters
        ----------
        X: list or tuple of arrays of shape (N, 2)
            The diagrams to fit the model to. Each diagram should be an array
            of shape (N, 2) for varying N, where each row of the array is the
            birth-death or birth-lifetime coordinates of a topological feature
            in the diagram. ``X`` should then be a list or tuple of such arrays.

        Returns
        -------
        vectors: array of shape (n_samples, n_components)
            The vectorization of the diagrams with a row of size ``n_components``
            for each diagram.
        """
        check_is_fitted(self, ["mixture_model_", "ground_distance_"])
        result = np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )

        if self.apply_umap:
            if self.umap_metric == "wasserstein":
                warn(
                    "Transform is not compatible with 'apply_umap=True'"
                    "and 'umap_metric=\"wasserstein\"'. Returning non-umap"
                    " diagram vectorization instead."
                )
                return result
            else:
                result = self.umap_.transform(result)

        return result

    def fit_transform(self, X: Sequence[np.ndarray], y=None, **kwargs) -> np.ndarray:
        """Fit a pervect model to the list of persistence diagrams X and
        return pervect representation of X.

        Parameters
        ----------
        X: list or tuple of arrays of shape (N, 2)
            The diagrams to fit the model to. Each diagram should be an array
            of shape (N, 2) for varying N, where each row of the array is the
            birth-death or birth-lifetime coordinates of a topological feature
            in the diagram. ``X`` should then be a list or tuple of such arrays.

        Returns
        -------
        vectors: array of shape (n_samples, n_components)
            The vectorization of the diagrams with a row of size ``n_components``
            for each diagram.
        """
        self.fit(X)
        if self.apply_umap:
            return self.umap_.embedding_
        else:
            return np.vstack(
                [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
            )

    def pairwise_p_wasserstein_distance(
        self, X: np.ndarray, p: Union[int, None] = None
    ) -> np.ndarray:
        """Compute (an approximation of) the all pairs p-Wasserstein distance
        between persistence diagrams X.

        Parameters
        ----------
        X: list or tuple of arrays of shape (N, 2)
            The diagrams to fit the model to. Each diagram should be an array
            of shape (N, 2) for varying N, where each row of the array is the
            birth-death or birth-lifetime coordinates of a topological feature
            in the diagram. ``X`` should then be a list or tuple of such arrays.

        p: int (optional, default=None)
            The p value to use when computing p-Wasserstein distance. If ``p``
            is ``None`` then the models p-value will be used.

        Returns
        -------
        distance_matrix: array of shape (n_diagrams, n_diagrams)
            The matrix of all pairwise p-Wasserstein distances between the
            persistence diagrams X.
        """
        if p is None:
            p = self.p
        vecs = self.transform(X)
        distance_matrix = pairwise_distances(
            vecs,
            metric=persistence_wasserstein_distance,
            ground_distance=self.ground_distance_ ** p,
        )
        return np.power(distance_matrix, 1.0 / p)
