from pervect import PersistenceVectorizer
from sklearn.utils.estimator_checks import (
    check_estimator,
    check_estimators_dtypes,
    check_fit_score_takes_y,
    check_dtype_object,
    check_pipeline_consistency,
    check_estimators_nan_inf,
    check_estimator_sparse_data,
    check_estimators_pickle,
    check_transformer_data_not_an_array,
    check_transformer_general,
    check_fit2d_predict1d,
    check_methods_subset_invariance,
    check_fit2d_1sample,
    check_fit2d_1feature,
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_fit_idempotent,
)
from sklearn.utils.validation import check_random_state
from sklearn.metrics import pairwise_distances
from pervect.pervect_ import (
    GaussianMixture,
    vectorize_diagram,
    wasserstein_diagram_distance,
    pairwise_gaussian_ground_distance,
    add_birth_death_line,
    persistence_wasserstein_distance,
)

import umap

import pytest

import numpy as np

np.random.seed(42)
base_data = np.vstack(
    [np.random.beta(1, 5, size=100), np.random.gamma(shape=0.5, scale=1.0, size=100),]
).T


def test_pervect_estimator():
    for estimator, check in check_estimator(PersistenceVectorizer, generate_only=True):
        # These all pass in unsuitable data, so skip them
        if check.func in (
            check_estimators_dtypes,
            check_fit_score_takes_y,
            check_dtype_object,
            check_pipeline_consistency,
            check_estimators_nan_inf,
            check_estimator_sparse_data,
            check_estimators_pickle,
            check_transformer_data_not_an_array,
            check_transformer_general,
            check_fit2d_predict1d,
            check_methods_subset_invariance,
            check_fit2d_1sample,
            check_fit2d_1feature,
            check_dict_unchanged,
            check_dont_overwrite_parameters,
            check_fit_idempotent,
        ):
            pass
        else:
            check(estimator)


def test_pervect_transform():

    random_seed = check_random_state(42)
    model = PersistenceVectorizer(n_components=4, random_state=random_seed).fit(
        base_data
    )
    model_result = model.transform(base_data)

    random_seed = check_random_state(42)
    gmm = GaussianMixture(n_components=4, random_state=random_seed).fit(base_data)
    util_result = np.array([vectorize_diagram(diagram, gmm) for diagram in base_data])

    assert np.allclose(model.mixture_model_.means_, gmm.means_)
    assert np.allclose(model.mixture_model_.covariances_, gmm.covariances_)
    assert np.allclose(model_result, util_result)

    random_seed = check_random_state(42)
    model_result = PersistenceVectorizer(
        n_components=4, random_state=random_seed
    ).fit_transform(base_data)

    assert np.allclose(model_result, util_result)

    random_seed = check_random_state(42)
    model = PersistenceVectorizer(
        n_components=4, random_state=random_seed, apply_umap=True,
    ).fit(base_data)
    model_result = model.transform(base_data)
    assert np.allclose(model.mixture_model_.means_, gmm.means_)
    assert np.allclose(model.mixture_model_.covariances_, gmm.covariances_)

    random_seed = check_random_state(42)
    umap_util_result = umap.UMAP(
        metric="hellinger", random_state=random_seed
    ).fit_transform(util_result)

    assert np.allclose(model_result, umap_util_result)

    random_seed = check_random_state(42)
    model = PersistenceVectorizer(
        n_components=4,
        random_state=random_seed,
        apply_umap=True,
        umap_metric="wasserstein",
    ).fit(base_data)
    model_result = model.umap_.embedding_

    precomputed_dmat = model.pairwise_p_wasserstein_distance(base_data)

    assert np.allclose(precomputed_dmat, model._distance_matrix)

    random_seed = check_random_state(42)
    umap_util_result = umap.UMAP(
        metric="precomputed", random_state=random_seed
    ).fit_transform(precomputed_dmat)
    assert np.allclose(model_result, umap_util_result)


def test_model_wasserstein():

    random_seed = check_random_state(42)
    model = PersistenceVectorizer(n_components=4, random_state=random_seed).fit(
        base_data
    )
    model_dmat = model.pairwise_p_wasserstein_distance(base_data[:10])

    random_seed = check_random_state(42)
    gmm = GaussianMixture(n_components=4, random_state=random_seed).fit(base_data)

    vec_data = [vectorize_diagram(base_data[i], gmm) for i in range(10)]
    raw_ground_distance = pairwise_gaussian_ground_distance(
        gmm.means_, gmm.covariances_,
    )
    ground_distance = add_birth_death_line(
        raw_ground_distance, gmm.means_, gmm.covariances_, y_axis="lifetime",
    )
    util_dmat = pairwise_distances(
                    vec_data,
                    metric=persistence_wasserstein_distance,
                    ground_distance=ground_distance,
                )

    assert np.allclose(model_dmat, util_dmat)


def test_bad_params():
    with pytest.raises(ValueError):
        PersistenceVectorizer(y_axis="bad").fit(base_data)
    with pytest.raises(ValueError):
        PersistenceVectorizer(apply_umap=True, umap_metric="bad").fit(base_data)
    with pytest.raises(ValueError):
        wasserstein_diagram_distance(
            base_data[0], base_data[1], y_axis="bad",
        )
    with pytest.raises(ValueError):
        PersistenceVectorizer().fit(np.ones((32, 32)))
    with pytest.raises(ValueError):
        PersistenceVectorizer().fit([np.ones((100, 4)) for i in range(5)])
    with pytest.raises(ValueError):
        PersistenceVectorizer(n_components="foo").fit(base_data)
    with pytest.raises(ValueError):
        PersistenceVectorizer(n_components=-1).fit(base_data)
    with pytest.raises(ValueError):
        PersistenceVectorizer(n_components="foo", apply_umap=True).fit(base_data)
    with pytest.raises(ValueError):
        PersistenceVectorizer(n_components=-1, apply_umap=True).fit(base_data)
    with pytest.warns(UserWarning):
        PersistenceVectorizer(umap_n_components=3).fit(base_data)
    with pytest.warns(UserWarning):
        PersistenceVectorizer(umap_metric="wasserstein").fit(base_data)
