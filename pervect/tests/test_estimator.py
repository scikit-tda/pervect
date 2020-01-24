
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

import pytest

import numpy as np

np.random.seed(42)
base_data = np.vstack(
    [
        np.random.beta(1, 5, size=100),
        np.random.gamma(shape=0.5, scale=1.0, size=100),
    ]
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

