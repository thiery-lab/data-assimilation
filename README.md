# Data assimilation in Python

Basic Python code for data assimilation inference methods and test models.

## Dependencies

Intended for use with Python 3. Environment with all dependencies can be set up using [conda](https://conda.io/miniconda.html) with

```
conda env create -f environment.yml
```

Alternatively `conda` or `pip` can be used to manually create a Python 3 environment. The following Python packages 

  * [NumPy](http://www.numpy.org/)
  * [SciPy](https://www.scipy.org/)
  * [POT: Python Optimal Transport](http://pot.readthedocs.io/en/stable/)
  * [PyFFTW](http://pyfftw.readthedocs.io/en/latest/)
  
are sufficient for using the inference methods and model classes implemented in the `dapy` package. To install in a `conda` environment run

```
conda install numpy scipy
conda install -c conda-forge pot
conda install -c conda-forge pyfftw
```

or using `pip`

```
pip install numpy scipy POT pyfttw
```

To run the Jupyter notebooks the following additional packages are required

   * [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html)
   * [Matplotlib](http://matplotlib.org/)
   * [Seaborn](http://seaborn.pydata.org/)
   
which can be installed with `conda` using

```
conda install jupyter matplotlib seaborn
```

or using `pip` with

```
pip install jupyter matplotlib seaborn
```

### Installing the `dapy` package

Once an environment with the required dependencies has been set up the `dapy` package should be installed in to it using the `setup.py` script. The package includes several [Cython](http://cython.org/) extension modules which are provided as both Cython and C source. To build the extensions (in place) using Cython (install with `conda install cython` or `pip install cython`) run

```
python setup.py build_ext -use-cython -use-cython-opts --inplace
```

This will build the extension modules directly from the Cython source and using Cython optimisations which give performance improvements at the cost of less safe array access.

Alternatively to build directly from the C source generated from the Cython files run

```
python setup.py build_ext --inplace
```

The `dapy` package can then be installed in to the current environment by running

```
python setup.py install
```

or to install in developer mode (sym-linking files rather than creating a hard copy)

```
python setup.py develop
```

