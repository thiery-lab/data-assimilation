# Data assimilation in Python

Python code for data assimilation inference methods and test models.

The models implemented include

  * a one-dimensional discrete-time model with non-linear dynamics commonly used as a particle filter test problem and originally proposed in [Netto et al. 1978](https://www.sciencedirect.com/science/article/pii/S1474667017661949),
  * a three-dimensional ordinary differential equation model with chaotic dynamics proposed in [Lorenz 1963](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2),
  * a toy atmospherical dynamics model on a one-dimensional spatial domain, commonly used as an example of spatio-temporal chaos in the data assimilation literature, proposed in [Lorenz 96](https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved),
  * a stochastically forced linear advection-diffusion partial differential equation model of turbulence on a one-dimensional periodic spatial domain from [Majda and Harlim 2012](http://www.cambridge.org/9781107016668),
  * a stochastically forced Kuramoto-Sivashinsky model, a fourth-order non-linear partial differential equation model on a one-dimensional spatial domain representing a toy model of instabilities in laminar wave fronts,
  * a stochastically forced Navier-Stokes fluid simulation on a periodic two-dimensional spatial domain.

The inference methods implemented include

  * exact Kalman filter (including linear operator and square-root based implementations),
  * ensemble Kalman filter (stochastic peturbed observation and deterministic square root implementations and localised variant),
  * bootstrap particle filter,
  * ensemble transform particle filter (global and localised variants).

Example usages of the models and inference methods are provided in the [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) notebooks in the `notebooks` directory.

## Dependencies

Intended for use with Python 3.6+. Environment with all dependencies can be set up using [conda](https://conda.io/miniconda.html) with

    conda env create -f environment.yml

Alternatively conda or [pip](https://pip.pypa.io/en/stable/) can be used to manually create a Python 3 environment. The minimal requirements for using the inference methods and model classes implemented in the `dapy` package are [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [PyFFTW](http://pyfftw.readthedocs.io/en/latest/) (for efficient FFT computations in models using spectral expansions) and [tqdm](https://github.com/tqdm) (for displaying progress bars during filtering). To install in a conda environment run

    conda install numpy scipy
    conda install -c conda-forge pyfftw tqdm

or using pip

    pip install numpy scipy pyfftw tqdm

The ensemble transport particle filter inference methods require solving optimal transport problems. A C++ extension module (written in Cython) wrapping a network simplex linear programming based exact solver is included in the `dapy.ot` sub-package. Alternatively if available, solvers from the [Python Optimal Transport](http://pot.readthedocs.io/en/stable/) library can be used. To install in the current environment run `conda install -c conda-forge pot` or `pip install POT`.

The example [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) notebooks includes plots produced using [Matplotlib](http://matplotlib.org/) and [Seaborn](http://seaborn.pydata.org/). To be able to run the notebooks locally the following additional packages should be installed with conda using

    conda install jupyter matplotlib seaborn

or using pip with

    pip install jupyter matplotlib seaborn


## Installing the `dapy` package

Once an environment with the required dependencies has been set up the `dapy` package should be installed in to it using the `setup.py` script. The package includes several [Cython](http://cython.org/) extension modules which are provided as both Cython and C / C++ source. To build the extensions directly from the C / C++ source files (which does not require Cython to be installed) run

```
python setup.py build_ext
```

To build the extensions using Cython (install with `conda install cython` or `pip install cython`) run

```
python setup.py build_ext --use-cython
```

This will build the extension modules directly from the Cython source files and using Cython optimisations which give performance improvements at the cost of less safe array access (these can be disabled with optional argument '--no-cython-opts').

The `dapy` package can then be installed in to the current environment by running

```
python setup.py install
```

or to install in developer mode (sym-linking files rather than creating a hard copy)

```
python setup.py develop
```
