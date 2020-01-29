from setuptools import setup


def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


configuration = {
    "name": "pervect",
    "version": "0.0.1",
    "description": "Persistence Diagram Vectorizer",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "persistence, persistent homology, diagram, vectorizer",
    "url": "http://github.com/lmcinnes/pynndescent",
    "author": "Leland McInnes",
    "author_email": "leland.mcinnes@gmail.com",
    "maintainer": "Leland McInnes",
    "maintainer_email": "leland.mcinnes@gmail.com",
    "license": "BSD",
    "packages": ["pervect"],
    "install_requires": [
        "scikit-learn >= 0.22",
        "scipy >= 1.0",
        "numba >= 0.46",
        "joblib >= 0.11",
        "pot >= 0.6",
        "umap-learn >= 0.3.10",
    ],
    "extras_require" : {  # use `pip install -e ".[testing]"``
                         'testing': [
                             'pytest',
                             'scipy'
                         ],
                         'docs': [  # `pip install -e ".[docs]"``
                             'sktda_docs_config'
                         ]
                     },
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": True,
}

setup(**configuration)
