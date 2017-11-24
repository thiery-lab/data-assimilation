"""Ensemble Kalman filters for inference in state space models."""

import numpy as np
import numpy.linalg as la


class EnsembleKalmanFilter(object):

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
        self.init_state_sampler = init_state_sampler
        self.next_state_sampler = next_state_sampler
        self.observation_sampler = observation_sampler
        self.rng = rng

    def filter(self, x_observed, n_particles):
        """Compute (approximate) filtering density parameters.

        Args:
            x_observed (array): Observed state sequence with shape
                `(n_steps, dim_x)` where `n_steps` is number of time steps in
                sequence and `dim_x` is dimensionality of observations.
            n_particles (integer): Number of particles to use to represent
                filtering distribution at each time step.

        Returns:
            Dictionary containing arrays of filtering density parameters -
                z_mean_seq: Array of filtering density means at all time steps.
                z_particles_seq: Array of particles representing filtering
                    distribution at each time step.
        """
        n_steps, dim_x = x_observed.shape
        for t in range(n_steps):
            # forecast update
            if t == 0:
                z_forecast = self.init_state_sampler(n_particles)
                dim_z = z_forecast.shape[1]
                z_mean_seq = np.full((n_steps, dim_z), np.nan)
                z_particles_seq = np.full(
                    (n_steps, n_particles, dim_z), np.nan)
            else:
                z_forecast = self.next_state_sampler(z_analysis, t-1)
            # analysis update
            x_forecast = self.observation_sampler(z_forecast, t)
            dz_forecast = z_forecast - z_forecast.mean(0)
            dx_forecast = x_forecast - x_forecast.mean(0)
            k_gain = np.linalg.pinv(dx_forecast).dot(dz_forecast)
            z_analysis = z_forecast + (x_observed[t] - x_forecast).dot(k_gain)
            z_mean_seq[t] = z_analysis.mean(0)
            z_particles_seq[t] = z_analysis
        return {'z_mean_seq': z_mean_seq, 'z_particles_seq': z_particles_seq}


class EnsembleSquareRootFilter(object):
    """Ensemble Kalman filter with deterministic matrix square root updates.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler()
    x[0] = H.dot(z[0]) + J.dot(v[0])
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = H.dot(z[t]) + J.dot(v[t])

    where

       z[t] : unobserved system state at time index t,
       x[t] : observed system state at time index t,
       v[t] : zero-mean identity covariance Gaussian observation noise
              vector at time index t,
       H: linear observation matrix,
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
        M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill, and J. S.
        Whitaker, Ensemble square root filters, Monthly Weather Review, 131
        (2003), pp. 1485--1490.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_matrix, obser_noise_matrix, rng):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_matrix (array): Matrix defining linear obervation
                operator.
            obser_noise_matrix (array): Matrix defining transformation of
                additive observation noise.
            rng (RandomState): Numpy RandomState random number generator.
        """
        self.init_state_sampler = init_state_sampler
        self.next_state_sampler = next_state_sampler
        self.observation_matrix = observation_matrix
        self.obser_noise_matrix = obser_noise_matrix
        self.obser_noise_covar = obser_noise_matrix.dot(obser_noise_matrix.T)
        self.rng = rng

    def filter(self, x_observed, n_particles, warn=True):
        """Compute (approximate) filtering density parameters.

        Args:
            x_observed (array): Observed state sequence with shape
                `(n_steps, dim_x)` where `n_steps` is number of time steps in
                sequence and `dim_x` is dimensionality of observations.
            n_particles (integer): Number of particles to use to represent
                filtering distribution at each time step.
            warn (boolean, default True): Warn if eigenvalues of matrix used
                to compute matrix square root for analysis perturbation
                ensemble update are outside of unit circle (eigenvalues are
                clipped to [-infty, 1] during update).

        Returns:
            Dictionary containing arrays of filtering density parameters -
                z_mean_seq: Array of filtering density means at all time steps.
                z_particles_seq: Array of particles representing filtering
                    distribution at each time step.
        """
        n_steps, dim_x = x_observed.shape
        for t in range(n_steps):
            if t == 0:
                z_forecast = self.init_state_sampler(n_particles)
                dim_z = z_forecast.shape[1]
                z_mean_seq = np.full((n_steps, dim_z), np.nan)
                z_particles_seq = np.full(
                    (n_steps, n_particles, dim_z), np.nan)
            else:
                z_forecast = self.next_state_sampler(z_analysis, t-1)
            z_mean_forecast = z_forecast.mean(0)
            x_mean_forecast = self.observation_matrix.dot(z_mean_forecast)
            dz_forecast = z_forecast - z_mean_forecast
            dx_forecast = dz_forecast.dot(self.observation_matrix.T)
            c_matrix = (
                dx_forecast.T.dot(dx_forecast) +
                (n_particles - 1) * self.obser_noise_covar
            )
            eigval_c, eigvec_c = la.eigh(c_matrix)
            k_gain = (eigvec_c / eigval_c).dot(eigvec_c.T).dot(
                dx_forecast.T.dot(dz_forecast))
            z_mean_analysis = z_mean_forecast + (
                x_observed[t] - x_mean_forecast).dot(k_gain)
            m_matrix = dx_forecast.dot(
                eigvec_c / eigval_c).dot(eigvec_c.T).dot(dx_forecast.T)
            eigval_m, eigvec_m = la.eigh(m_matrix)
            if warn and np.any(eigval_m > 1.):
                print('Warning: eigenvalue(s) outside unit circle, max: {0}'
                      .format(eigval_m.max()))
            sqrt_matrix = (
                eigvec_m * abs(1 - np.clip(eigval_m, -np.inf, 1.))**0.5
            ).dot(eigvec_m.T)
            dz_analysis = sqrt_matrix.dot(dz_forecast)
            z_analysis = z_mean_analysis + dz_analysis
            z_particles_seq[t] = z_analysis
            z_mean_seq[t] = z_mean_analysis
        return {'z_mean_seq': z_mean_seq, 'z_particles_seq': z_particles_seq}


class WoodburyEnsembleSquareRootFilter(object):
    """Ensemble Kalman filter with deterministic matrix square root updates.

    Uses Woodbury identity to compute matrix inverse using explicit inverse
    of observation noise covariance to avoid O(dim_x**3) operations where
    `dim_x` is the observation vector dimensionality.

    Assumes the system dynamics are of the form

    z[0] = init_state_sampler()
    x[0] = H.dot(z[0]) + J.dot(v[0])
    for t in range(1, T):
        z[t] = next_state_sampler(z[t-1], t-1)
        x[t] = H.dot(z[t]) + J.dot(v[t])

    where

       z[t] : unobserved system state at time index t,
       x[t] : observed system state at time index t,
       v[t] : zero-mean identity covariance Gaussian observation noise
              vector at time index t,
       H: linear observation matrix,
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
        M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill, and J. S.
        Whitaker, Ensemble square root filters, Monthly Weather Review, 131
        (2003), pp. 1485--1490.
    """

    def __init__(self, init_state_sampler, next_state_sampler,
                 observation_matrix, obser_noise_preci, rng):
        """
        Args:
            init_state_sampler (function): Function returning sample(s) from
                initial state distribution. Takes number of particles to sample
                as argument.
            next_state_sampler (function): Function returning sample(s) from
                distribution on next state given current state(s). Takes array
                of current state(s) and current time index as
                arguments.
            observation_matrix (array): Matrix defining linear obervation
                operator.
            obser_noise_preci (array): Matrix defining precision of additive
                Gaussian observation noise (inverse of covariance matrix).
            rng (RandomState): Numpy RandomState random number generator.
        """
        self.init_state_sampler = init_state_sampler
        self.next_state_sampler = next_state_sampler
        self.observation_matrix = observation_matrix
        self.obser_noise_preci = obser_noise_preci
        self.rng = rng

    def filter(self, x_observed, n_particles, warn=True):
        """Compute (approximate) filtering density parameters.

        Args:
            x_observed (array): Observed state sequence with shape
                `(n_steps, dim_x)` where `n_steps` is number of time steps in
                sequence and `dim_x` is dimensionality of observations.
            n_particles (integer): Number of particles to use to represent
                filtering distribution at each time step.
            warn (boolean, default True): Warn if eigenvalues of matrix used
                to compute matrix square root for analysis perturbation
                ensemble update are outside of unit circle (eigenvalues are
                clipped to [-infty, 1] during update).

        Returns:
            Dictionary containing arrays of filtering density parameters -
                z_mean_seq: Array of filtering density means at all time steps.
                z_particles_seq: Array of particles representing filtering
                    distribution at each time step.
        """
        n_steps, dim_x = x_observed.shape
        for t in range(n_steps):
            if t == 0:
                z_forecast = self.init_state_sampler(n_particles)
                dim_z = z_forecast.shape[1]
                z_mean_seq = np.full((n_steps, dim_z), np.nan)
                z_particles_seq = np.full(
                    (n_steps, n_particles, dim_z), np.nan)
            else:
                z_forecast = self.next_state_sampler(z_analysis, t-1)
            z_mean_forecast = z_forecast.mean(0)
            x_mean_forecast = self.observation_matrix.dot(z_mean_forecast)
            dz_forecast = z_forecast - z_mean_forecast
            dx_forecast = dz_forecast.dot(self.observation_matrix.T)
            obs_precis_dx_forecast = self.obser_noise_preci.dot(dx_forecast.T)
            c_inv_matrix = (
                self.obser_noise_preci -
                obs_precis_dx_forecast.dot(
                    la.solve(
                        (n_particles - 1) * np.eye(n_particles) +
                        dx_forecast.dot(obs_precis_dx_forecast),
                        obs_precis_dx_forecast.T,
                    )
                )
            ) / (n_particles - 1)
            c_inv_dx_forecast_t = c_inv_matrix.dot(dx_forecast.T)
            k_gain = c_inv_dx_forecast_t.dot(dz_forecast)
            z_mean_analysis = z_mean_forecast + (
                x_observed[t] - x_mean_forecast).dot(k_gain)
            m_matrix = dx_forecast.dot(c_inv_dx_forecast_t)
            eigval_m, eigvec_m = la.eigh(m_matrix)
            if warn and np.any(eigval_m > 1.):
                print('Warning: eigenvalue(s) outside unit circle, max: {0}'
                      .format(eigval_m.max()))
            sqrt_matrix = (
                eigvec_m * (1 - np.clip(eigval_m, -np.inf, 1.))**0.5
            ).dot(eigvec_m.T)
            dz_analysis = sqrt_matrix.dot(dz_forecast)
            z_analysis = z_mean_analysis + dz_analysis
            z_particles_seq[t] = z_analysis
            z_mean_seq[t] = z_mean_analysis
        return {'z_mean_seq': z_mean_seq, 'z_particles_seq': z_particles_seq}
