"""Ensemble Kalman filters for inference in state space models."""

import numpy as np
import numpy.linalg as la
from dapy.models.base import inherit_docstrings
from dapy.inference.base import (
        AbstractEnsembleFilter, AbstractLocalEnsembleFilter)


@inherit_docstrings
class EnsembleKalmanFilter(AbstractEnsembleFilter):
    """Ensemble Kalman filter with perturbed observations.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler(rng)
    x[0] = observation_sampler(z[0], 0)
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = observation_sampler(z[t], t)

    where

       z[t] : unobserved system state at time index t,
       x[t] : observed system state at time index t,
       init_state_sampler: function sampling from initial state distribution,
       observation_sampler: function sampling from distribution of observed
           state at a time index given unoberved state at this time index,
       next_state_sampler: function sampling state at current time index given
           state at previous time index, describing system dynamics.

    The distribution of the system state at each time step is approximated
    by propagating a system of particles each representing a single system
    forward through time. For a model with linear-Gaussian dynamics in the
    limit of an infinite number of particles the ensemble Kalman filter
    converges to giving exact results.

    References:
        G. Evensen, Sequential data assimilation with nonlinear
        quasi-geostrophic model using Monte Carlo methods to forecast error
        statistics, Journal of Geophysical Research, 99 (C5) (1994), pp.
        143--162

        P. Houtekamer and H. L. Mitchell, Data assimilation using an ensemble
        Kalman filter technique, Monthly Weather Review, 126 (1998), pp.
        796--811
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_sampler, rng):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_sampler (function): Function returning sample(s) from
                distribution on observations given current state(s). Takes
                array of current state(s) and current time index as arguments.
            rng (RandomState): Numpy RandomState random number generator.
        """
        super(EnsembleKalmanFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler, rng=rng
        )
        self.observation_sampler = observation_sampler

    def analysis_update(self, z_forecast, x_observed, time_index):
        x_forecast = self.observation_sampler(z_forecast, time_index)
        dz_forecast = z_forecast - z_forecast.mean(0)
        dx_forecast = x_forecast - x_forecast.mean(0)
        dx_error = x_observed - x_forecast
        z_analysis = z_forecast + (
            dx_error.dot(la.pinv(dx_forecast))).dot(dz_forecast)
        return z_analysis, z_analysis.mean(0), z_analysis.std(0)


@inherit_docstrings
class EnsembleSquareRootFilter(AbstractEnsembleFilter):
    """Ensemble Kalman filter with deterministic matrix square root updates.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler()
    x[0] = H.dot(z[0]) + J.dot(v[0])
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = observation_func(z[t], t) + J.dot(v[t])

    where

       z[t] : unobserved system state at time index t,
       x[t] : observed system state at time index t,
       v[t] : zero-mean identity covariance Gaussian observation noise
              vector at time index t,
       observation_func: function defining observation operator,
       J: observation noise transform matrix,
       init_state_sampler: function sampling from initial state distribution,
       next_state_sampler: function sampling state at current time index given
           state at previous time index, describing system dynamics.

    The distribution of the system state at each time step is approximated
    by propagating a system of particles each representing a single system
    forward through time. For a model with linear-Gaussian dynamics in the
    limit of an infinite number of particles the ensemble Kalman filter
    converges to giving exact results.

    References:
        1. M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill,
           and J. S. Whitaker, Ensemble square root filters,
           Monthly Weather Review, 131 (2003), pp. 1485--1490.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_matrix, rng, warn=True):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_func (function): Function returning pre-noise
                observations given current state(s). Takes array of current
                state(s) and current time index as arguments.
            obser_noise_matrix (array): Matrix defining transformation of
                additive observation noise.
            rng (RandomState): Numpy RandomState random number generator.
            warn (boolean, default True): Warn if eigenvalues of matrix used
                to compute matrix square root for analysis perturbation
                ensemble updates are outside of unit circle (eigenvalues are
                clipped to [-infty, 1] during filtering updates).
        """
        super(EnsembleSquareRootFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler, rng=rng
        )
        self.observation_func = observation_func
        self.obser_noise_matrix = obser_noise_matrix
        self.obser_noise_covar = obser_noise_matrix.dot(obser_noise_matrix.T)
        self.warn = warn

    def analysis_update(self, z_forecast, x_observed, time_index):
        n_particles = z_forecast.shape[0]
        z_mean_forecast = z_forecast.mean(0)
        dz_forecast = z_forecast - z_mean_forecast
        x_forecast = self.observation_func(z_forecast, time_index)
        x_mean_forecast = x_forecast.mean(0)
        dx_forecast = x_forecast - x_mean_forecast
        c_matrix = (
            dx_forecast.T.dot(dx_forecast) +
            (n_particles - 1) * self.obser_noise_covar)
        eigval_c, eigvec_c = la.eigh(c_matrix)
        k_gain = (eigvec_c / eigval_c).dot(eigvec_c.T).dot(
            dx_forecast.T.dot(dz_forecast))
        z_mean_analysis = z_mean_forecast + (
            x_observed - x_mean_forecast).dot(k_gain)
        m_matrix = dx_forecast.dot(
            eigvec_c / eigval_c).dot(eigvec_c.T).dot(dx_forecast.T)
        eigval_m, eigvec_m = la.eigh(m_matrix)
        if self.warn and np.any(eigval_m > 1.):
            print('Warning: eigenvalue(s) outside unit circle, max: {0}'
                  .format(eigval_m.max()))
        sqrt_matrix = (
            eigvec_m * abs(1 - np.clip(eigval_m, -np.inf, 1.))**0.5
        ).dot(eigvec_m.T)
        dz_analysis = sqrt_matrix.dot(dz_forecast)
        return (z_mean_analysis + dz_analysis, z_mean_analysis,
                (dz_analysis**2).mean(0)**0.5)


@inherit_docstrings
class WoodburyEnsembleSquareRootFilter(AbstractEnsembleFilter):
    """Ensemble Kalman filter with deterministic matrix square root updates.

    Uses Woodbury identity to compute matrix inverse using explicit inverse
    of observation noise covariance to avoid O(dim_x**3) operations where
    `dim_x` is the observation vector dimensionality.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler()
    x[0] = H.dot(z[0]) + J.dot(v[0])
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = observation_func(z[t], t) + J.dot(v[t])

    where

       z[t] : unobserved system state at time index t,
       x[t] : observed system state at time index t,
       v[t] : zero-mean identity covariance Gaussian observation noise
              vector at time index t,
       observation_func: function defining observation operator,
       J: observation noise transform matrix,
       init_state_sampler: function sampling from initial state distribution,
       next_state_sampler: function sampling state at current time index given
           state at previous time index, describing system dynamics.

    The distribution of the system state at each time step is approximated
    by propagating a system of particles each representing a single system
    forward through time. For a model with linear-Gaussian dynamics in the
    limit of an infinite number of particles the ensemble Kalman filter
    converges to giving exact results.

    References:
        1. M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill,
           and J. S. Whitaker, Ensemble square root filters,
           Monthly Weather Review, 131 (2003), pp. 1485--1490.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_preci, rng, warn=True):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as arguments.
            observation_func (function): Function returning pre-noise
                observations given current state(s). Takes array of current
                state(s) and current time index as arguments.
            obser_noise_preci (array): Matrix defining precision of additive
                Gaussian observation noise (inverse of covariance matrix).
            rng (RandomState): Numpy RandomState random number generator.
            warn (boolean, default True): Warn if eigenvalues of matrix used
                to compute matrix square root for analysis perturbation
                ensemble updates are outside of unit circle (eigenvalues are
                clipped to [-infty, 1] during filtering updates).
        """
        super(WoodburyEnsembleSquareRootFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler, rng=rng
        )
        self.observation_func = observation_func
        self.obser_noise_preci = obser_noise_preci
        self.warn = warn

    def analysis_update(self, z_forecast, x_observed, time_index):
        n_particles = z_forecast.shape[0]
        z_mean_forecast = z_forecast.mean(0)
        dz_forecast = z_forecast - z_mean_forecast
        x_forecast = self.observation_func(z_forecast, time_index)
        x_mean_forecast = x_forecast.mean(0)
        dx_forecast = x_forecast - x_mean_forecast
        dx_error = x_observed - x_mean_forecast
        a_matrix = self.obser_noise_preci.dot(dx_forecast.T)
        b_vector = dx_error.dot(a_matrix)
        c_matrix = dx_forecast.dot(a_matrix)
        d_matrix = (n_particles - 1) * np.eye(n_particles) + c_matrix
        e_matrix = la.solve(d_matrix, c_matrix)
        z_mean_analysis = z_mean_forecast + (
            (b_vector - b_vector.dot(e_matrix)).dot(dz_forecast) /
            (n_particles - 1))
        m_matrix = (c_matrix - c_matrix.dot(e_matrix)) / (n_particles - 1)
        eigval_m, eigvec_m = la.eigh(m_matrix)
        if self.warn and np.any(eigval_m > 1.):
            print('Warning: eigenvalue(s) outside unit circle, max: {0}'
                  .format(eigval_m.max()))
        sqrt_matrix = (
            eigvec_m * (1 - np.clip(eigval_m, -np.inf, 1.))**0.5
        ).dot(eigvec_m.T)
        dz_analysis = sqrt_matrix.dot(dz_forecast)
        return (z_mean_analysis + dz_analysis, z_mean_analysis,
                (dz_analysis**2).mean(0)**0.5)


@inherit_docstrings
class LocalEnsembleTransformKalmanFilter(AbstractLocalEnsembleFilter):
    """Localised ensemble transform Kalman filter for spatially extended models

    References:
        1. Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007).
           Efficient data assimilation for spatiotemporal chaos:
           A local ensemble transform Kalman filter.
           Physica D: Nonlinear Phenomena, 230(1), 112-126.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_func, obser_noise_std, n_grid, localisation_func,
                 inflation_factor, rng):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_func (function): Function returning pre-noise
                observations given current state(s). Takes array of current
                state(s) and current time index as arguments.
            obser_noise_std (array): One-dimensional array defining standard
                deviations of additive Gaussian observation noise on each
                dimension with it assumed that the noise is independent across
                dimensions i.e. a diagonal observation noise covariance matrix.
            n_grid (integer): Number of spatial points over which state is
                defined. Typically points will be on a rectilinear grid though
                this is not actually required. It is assumed that if `z` is a
                state vector of size `dim_z` then `dim_z % n_grid == 0` and
                that `z` is ordered such that iterating over the last
                dimension of a reshaped array
                    z_grid = z.reshape((dim_z // n_grid, n_grid))
                will correspond to iterating over the state component values
                across the different spatial (grid) locations.
            localisation_func (function): Function (or callable object) which
                given an index corresponding to a spatial grid point (i.e.
                the iteration index over the last dimension of a reshaped
                array `z_grid` as described above) will return an array of
                integer indices into an observation vector and corresponding
                array of weight coefficients specifying the observation vector
                entries 'local' to state grid point described by the index and
                there corresponding weights (with closer observations
                potentially given larger weights).
            inflation_factor (float): A value greater than or equal to one used
                to inflate the analysis ensemble on each update as a heuristic
                to overcome the underestimation of the uncertainty in the
                system state by ensemble Kalman filter methods.
            rng (RandomState): Numpy RandomState random number generator.
        """
        super(LocalEnsembleTransformKalmanFilter, self).__init__(
                init_state_sampler=init_state_sampler,
                next_state_sampler=next_state_sampler,
                observation_func=observation_func,
                obser_noise_std=obser_noise_std,
                n_grid=n_grid, localisation_func=localisation_func, rng=rng
        )
        self.inflation_factor = inflation_factor

    def local_analysis_update(self, z_forecast, x_forecast, x_observed,
                              obs_noise_std, localisation_weights):
        n_particles = z_forecast.shape[0]
        z_mean_forecast = z_forecast.mean(0)
        dz_forecast = z_forecast - z_mean_forecast
        x_mean_forecast = x_forecast.mean(0)
        dx_forecast = x_forecast - x_mean_forecast
        c_matrix = ((dx_forecast / obs_noise_std) * localisation_weights)
        p_inv_matrix = (
            (n_particles - 1) * np.eye(n_particles) / self.inflation_factor +
            c_matrix.dot(dx_forecast.T))
        eigval_p_inv, eigvec_p_inv = la.eigh(p_inv_matrix)
        d_vector = c_matrix.dot(x_observed - x_mean_forecast)
        dw_matrix = (n_particles - 1)**0.5 * eigvec_p_inv / eigval_p_inv**0.5
        w_mean = (eigvec_p_inv / eigval_p_inv).dot(
            eigvec_p_inv.T.dot(d_vector))
        w_matrix = w_mean + dw_matrix
        return z_mean_forecast + w_matrix.dot(dz_forecast)
