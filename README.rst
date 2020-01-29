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

Vectorization of persistence diagrams and approximate Wasserstein distance. This is
managed by approximating persistence diagrams with Gaussian mixture models and then
measuring the Wasserstein distance between the Gaussian mixtures. As the number of
components in mixture model increases the accuracy of the approximation increases
accordingly until, with equivalence in the limit.

The library is implemented as a `Scikit-learn <https://scikit-learn.org/stable/>`_
transformer -- taking a list of
persistence diagrams (preferably in birth-lifetime format) as input, and transforming
it into a vector representation (specifically the component weights for a Gaussian
mixture model fit to the union of all the diagrams). Distances can then be computed
as Wassterstein distance over a ground-distance matrix provided as an attribute of the
transformer. Alternatively UMAP can be used to convert toa lower dimensional
Euclidean distance representation.

------------------
How to use PerVect
------------------

The pervect library inheritis from sklearn classes and can be used as an sklearn
transformer.

.. code:: python

    import pervect
    vects = pervect.PersistenceVectorizer().fit_transform(diagrams)

It can also be used in standard sklearn pipelines along with other machine learning
tools including clustering and classifiers.

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


