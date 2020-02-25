.. image:: https://badge.fury.io/py/pervect.svg
    :target: https://pypi.org/project/pervect/0.0.1/
.. image:: https://travis-ci.org/scikit-tda/pervect.svg?branch=master
    :target: https://travis-ci.org/scikit-tda/pervect
.. image:: https://codecov.io/gh/scikit-tda/pervect/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/scikit-tda/pervect
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause

=======
PerVect
=======

PerVect is a library for Persistence-diagram Vectorization -- converting the
output
of a persistent homology computation to a vector from which it is still possible to
compute a close approximation to persistent Wasserstein distance. This is
managed by approximating a training set of persistence diagrams with Gaussian mixture
models; vectorizing a diagram as the weighted maximum likelihood estimate of the
mixture weights for the learned components given the diagram; and then measuring the
Wasserstein distance between vectorized diagrams by the Wasserstein distance between
the corresponding Gaussian mixtures. As the number of
components in mixture model increases the accuracy of the approximation increases
accordingly, with equivalence in the limit.

The library is implemented as a `Scikit-learn <https://scikit-learn.org/stable/>`_
transformer -- taking a list of
persistence diagrams (preferably in birth-lifetime format) as input, and producing
vector representations. Alternatively UMAP can be used to convert to a lower dimensional
Euclidean distance representation.

------------
Installation
------------

Requirements:

* Python >= 3.6
* scikit-learn
* umap-learn
* numba
* joblib
* pot

You can install pervect from PyPI with pip:

.. code:: bash

    pip install pervect
    
---------------------
User Guide / Tutorial
---------------------

.. toctree::
   :maxdepth: 2
   
   basic_usage
   clustering_persistence_diagrams
   wasserstein_distances
   
----------
Background
----------

.. toctree::
   :maxdepth: 2

   how_pervect_works
   performance_and_scalability

-------------
API Reference
-------------

.. toctree::

   api
