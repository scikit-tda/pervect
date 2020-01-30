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

------------------
How to use PerVect
------------------

The pervect library inherits from sklearn classes and can be used as an sklearn
transformer. Assuming that you have a list persistence diagrams where each
diagram is a numpy array of points in 2D then you can vectorize by simply applying:

.. code:: python

    import pervect
    vects = pervect.PersistenceVectorizer().fit_transform(diagrams)

It can also be used in standard sklearn pipelines along with other machine learning
tools including clustering and classifiers. For example, given a set of training
diagrams, and a separate test set of diagrams we could do:

.. code:: python

   import pervect
   vectorizer = pervect.PersistenceVectorizer().fit(train)
   train_vectors = vectorizer.transform(train)
   test_vectors = vectorizer.transform(test)

The vectorizer is also effective at efficiently approximating Wasserstein distance
between diagrams. A trained model can compute pairwise Wasserstein distance between
a list of diagrams as follows:

.. code:: python

   import pervect
   vectorizer = pervect.PersistenceVectorizer().fit(train)
   test_diagram_distances = vectorizer.pairwise_p_wasserstein_distance(test, p=1)

The vectorizer can also automatically produce UMAP representations of the diagrams,
either using "hellinger" distance or Wasserstein distance (note that transforming
new data using Wassersteing trained UMAP is currently unavailable).

.. code:: python

   import pervect
   diagram_map = pervect.PersistenceVectorizer(apply_umap=True).fit(diagrams)


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

For a manual install get this package:

.. code:: bash

    wget https://github.com/scikit-tda/pervect/archive/master.zip
    unzip master.zip
    rm master.zip
    cd pervect-master

Install the requirements

.. code:: bash

    sudo pip install -r requirements.txt

Install the package

.. code:: bash

    pip install .

----------
References
----------

This package was inspired by and builds upon the work of Elizabeth Munch, Jose Perea,
Firas Khasawneh and Sarah Tymochko. You can refer the the papers:

Jose A. Perea, Elizabeth Munch, Firas A. Khasawneh, *Approximating Continuous
Functions on Persistence Diagrams Using Template Functions*, arXiv:1902.07190

Sarah Tymochko, Elizabeth Munch, Firas A. Khasawneh, *Adaptive Partitioning for
Template Functions on Persistence Diagrams*, arXiv:1910.08506v1

-------
License
-------

The pervect package is 3-clause BSD licensed.

We would like to note that the pervect package makes heavy use of
NumFOCUS sponsored projects, and would not be possible without
their support of those projects, so please `consider contributing to NumFOCUS <https://www.numfocus.org/membership>`_.

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. To contribute please
`fork the project <https://github.com/scikit-tda/pervect/issues#fork-destination-box>`_
make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.


