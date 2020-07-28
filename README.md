# Data assimilation in Python

Python code for data assimilation inference methods and test models.

The models implemented include

  * a one-dimensional discrete-time model with non-linear dynamics commonly used as a particle filter test problem and originally proposed by Netto et al. (1978),
  * a three-dimensional ordinary differential equation model with chaotic dynamics proposed by Lorenz (1963),
  * a toy atmospherical dynamics model on a one-dimensional spatial domain, commonly used as an example of spatio-temporal chaos in the data assimilation literature, proposed by Lorenz (1996),
  * a linear advection-diffusion stochastic partial differential equation model of turbulence on a one-dimensional periodic spatial domain proposed by Majda and Harlim (2012),
  * a non-linear stochastic partial differential equation model on a one-dimensional spatial domain based on the the Kuramoto-Sivashinsky equation (Kuramoto and Tsuzuki, 1976; Sivashinsky, 1997), a fourth-order non-linear partial differential equation model of instabilities in laminar wave fronts,
  * a non-linear Navier-Stokes stochastic partial differential equation model on a periodic two-dimensional spatial domain.

The inference methods implemented include

  * Kalman filter (Kalman, 1960),
  * stochastic (perturbed observation) ensemble Kalman filter (Evensen, 1994; Bugers, van Leeuwen and Evensen, 1998),
  * ensemble transform (square-root) Kalman filter (Tippett et al., 2003),
  * local ensemble transform Kalman filter (Hunt, Kostelich and Szunyogh, 2007),
  * bootstrap particle filter (Gordon, Salmond and Smith, 1993),
  * ensemble transform particle filter (Reich, 2013),
  * local ensemble transform particle filter (Cheng and Reich, 2015),
  * scalable local ensemble transform particle filter (Graham and Thiery, 2019).

Example usages of the models and inference methods are provided in the [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) notebooks in the [`notebooks` directory](https://nbviewer.jupyter.org/github/thiery-lab/data-assimilation/tree/master/notebooks/).

## Dependencies

Intended for use with Python 3.6+. Environment with all dependencies can be set up using [conda](https://conda.io/miniconda.html) with

    conda env create -f environment.yml

Alternatively conda or [pip](https://pip.pypa.io/en/stable/) can be used to manually create a Python 3 environment. The minimal requirements for using the inference methods and model classes implemented in the `dapy` package are [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), (Numba)[http://numba.pydata.org/] and [PyFFTW](http://pyfftw.readthedocs.io/en/latest/) (for efficient FFT computations in models using spectral expansions). To install in a conda environment run

    conda install numpy scipy numba
    conda install -c conda-forge pyfftw

or using pip

    pip install numpy scipy numba pyfftw

The ensemble transport particle filter inference methods require solving optimal transport problems. A C++ extension module (written in Cython) wrapping a network simplex linear programming based exact solver is included in the `dapy.ot` sub-package. Alternatively if available, solvers from the [Python Optimal Transport](http://pot.readthedocs.io/en/stable/) library can be used. To install in the current environment run `conda install -c conda-forge pot` or `pip install POT`.

The example [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) notebooks includes plots produced using [Matplotlib](http://matplotlib.org/). To be able to run the notebooks locally the following additional packages should be installed with conda using

    conda install jupyter matplotlib

or using pip with

    pip install jupyter matplotlib


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

## References

  1.  Netto, M. A., Gimeno, L. and Mendes, M. J. (1978).
      A new spline algorithm for non-linear filtering of discrete time systems.
      IFAC Proceedings Volumes, 11(1), 2123-2130.
  2.  Lorenz, E. N. (1963).
      Deterministic nonperiodic flow.
      Journal of the atmospheric sciences, 20(2), 130-141.
  3.  Lorenz, E. N. (1996).
      Predictability - A problem partly solved.
      In Proceedings of Seminar on Predictability (1). European Centre for Medium-Range Weather Forecasts.
  4.  Majda, A. J. and Harlim, J. (2012).
      Filtering complex turbulent systems.
      Cambridge University Press.
  5.  Kuramoto, Y. and Tsuzuki, T. (1976).
      Persistent propagation of concentration waves indissipative media far from thermal equilibrium.
      Progress of theoretical physics (55), pp. 356--369.
  6.  Sivashinsky, G. (1977).
      Nonlinear analysis of hydrodynamic instability in laminar flames -- I. Derivation of basic equations.
      Acta Astronautica (4), pp. 1177--1206.
  7.  Kalman, R. E. (1960).
      A new approach to linear filtering and prediction problems.
      Transactions of the ASME -- Journal of Basic Engineering,
      Series D, 82, pp. 35--45.
  8.  Evensen, G. (1994).
      Sequential data assimilation with nonlinear quasi-geostrophic model
      using Monte Carlo methods to forecast error statistics.
      Journal of Geophysical Research, 99 (C5), pp. 143--162
  9.  Burgers, G.,van Leeuwen, P. J. and Evensen, G. (1998).
      Analysis scheme in the ensemble Kalman filter.
      Monthly Weather Review, (126) pp 1719--1724.
  10. Tippett,  M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M. and Whitaker, J. S. (2003).
      Ensemble square root filters.
      Monthly Weather Review, 131, pp. 1485--1490.
  11. Hunt, B. R., Kostelich, E. J. and Szunyogh, I. (2007).
      Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter.
      Physica D: Nonlinear Phenomena, 230(1), 112-126.
  12. Gordon, N.J., Salmond, D.J. and Smith, A.F.M. (1993).
      Novel approach to nonlinear / non-Gaussian Bayesian state estimation.
      Radar and Signal Processing, IEE Proceedings F. 140 (2): 107--113.
  13. Reich, S. (2013).
      A nonparametric ensemble transform method for Bayesian inference.
      SIAM Journal on Scientific Computing, 35(4), A2013-A2024.
  14. Cheng, Y. and Reich, S. (2015).
      Assimilating data into scientific models: An optimal coupling perspective.
      In Nonlinear Data Assimilation, pp 75--118. Springer.
  15. Graham, M. M. and Thiery A. H. (2019).
      A scalable optimal transport based particle filter.
      arXiv preprint 1906.00507.
